"""Timestamped public news, official agency release, and SEC EDGAR constituent event collector.

Enforces strict verification, anti-lookahead (`published_at_utc <= cutoff_utc`),
horizon freshness (24h intraday, 72h 1-3d swing), primary-vs-secondary distinction,
repost deduplication, and explicit symbol/channel impact mapping. Unverified items
remain `unverified_unknown` and never steer directional judgment.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import hashlib
import html as html_lib
import json
import re
from typing import Any
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import httpx

ALLOWED_NEWS_HOSTS = {
    "www.federalreserve.gov",
    "federalreserve.gov",
    "www.bls.gov",
    "bls.gov",
    "www.bea.gov",
    "bea.gov",
    "data.sec.gov",
    "www.sec.gov",
    "sec.gov",
    "home.treasury.gov",
    "nvidianews.nvidia.com",
    "blogs.nvidia.com",
    "www.apple.com",
    "apple.com",
    "blogs.microsoft.com",
    "news.microsoft.com",
    "blog.google",
    "about.fb.com",
}
PRIMARY_OFFICIAL_HOSTS = ALLOWED_NEWS_HOSTS | {
    "treasury.gov",
    "imf.org",
    "www.imf.org",
    "bis.org",
    "www.bis.org",
    "nvidia.com",
    "www.nvidia.com",
    "microsoft.com",
    "google.com",
    "about.google",
    "meta.com",
    "about.meta.com",
    "fb.com",
}

RSS_FEEDS = (
    ("fed_monetary", "Federal Reserve Monetary Policy", "https://www.federalreserve.gov/feeds/press_monetary.xml"),
    ("fed_speeches", "Federal Reserve Speeches", "https://www.federalreserve.gov/feeds/speeches.xml"),
    ("bls_latest", "US Bureau of Labor Statistics", "https://www.bls.gov/feed/bls_latest.rss"),
    ("bea_news", "US Bureau of Economic Analysis", "https://www.bea.gov/rss/rss.xml"),
)

CORPORATE_IR_FEEDS = (
    ("nvda_newsroom", "NVIDIA Official Newsroom", "https://nvidianews.nvidia.com/releases.xml"),
    ("aapl_newsroom", "Apple Newsroom", "https://www.apple.com/newsroom/rss-feed.rss"),
    ("msft_newsroom", "Microsoft Official Blog", "https://blogs.microsoft.com/feed/"),
    ("googl_newsroom", "Google Official Blog", "https://blog.google/rss/"),
    ("meta_newsroom", "Meta Newsroom", "https://about.fb.com/news/feed/"),
)

# Top Nasdaq-100 constituents monitored via official SEC EDGAR public JSON submissions.
NDX_TOP_CIKS: tuple[tuple[str, str], ...] = (
    ("NVDA", "0001045810"),
    ("AAPL", "0000320193"),
    ("MSFT", "0000789019"),
    ("AMZN", "0001018724"),
    ("META", "0001326801"),
    ("GOOGL", "0001652044"),
)
# Compliant SEC EDGAR declarative User-Agent (AppName email)
EDGAR_USER_AGENT = "TradingLabResearch research-contact@example.org"

_SECONDARY_RE = re.compile(
    r"\b(?:opinion|commentary|analysis|editorial|recap|roundup|wrap|preview|column|newsletter|opinionated)\b|解读|评论|综述|转引|预测汇总",
    re.I,
)
_MACRO_RE = re.compile(
    r"\b(?:fomc|federal reserve|fed|powell|cpi|pce|ppi|inflation|nonfarm|payroll|employment|unemployment|gdp|treasury|yield|rate hike|rate cut|monetary policy|beige book)\b|通胀|美联储|非农|利率|降息|加息|国债",
    re.I,
)
_GEO_RE = re.compile(
    r"\b(?:tariff|sanction|geopolitical|conflict|strike|escalation|shutdown|debt ceiling|central bank gold|safe[- ]haven)\b|关税|制裁|地缘|冲突|停摆|避险|央行购金",
    re.I,
)
_BTC_RE = re.compile(
    r"\b(?:bitcoin|btc|spot bitcoin etf|bitcoin etf|crypto|digital asset|stablecoin|sec|cftc|ibit|fbit|grayscale)\b|比特币|现货etf|数字资产|加密监管",
    re.I,
)
_NDX_RE = re.compile(
    r"\b(?:nasdaq|nvidia|nvda|apple|aapl|microsoft|msft|amazon|amzn|meta|alphabet|googl|ai chip|semiconductor|8-k|10-q|10-k|earnings|guidance|antitrust)\b|纳指|纳斯达克|英伟达|苹果|微软|亚马逊|财报|半导体",
    re.I,
)
_GOLD_RE = re.compile(
    r"\b(?:gold|bullion|xau|central bank reserves|precious metal)\b|黄金|金价|现货黄金",
    re.I,
)
_HAWKISH_RE = re.compile(
    r"\b(?:rate hike|hawkish|higher for longer|hotter inflation|inflation accelerates|tightening|sticky inflation|yields surge|beats estimates sharply|upside inflation surprise)\b|鹰派|加息|通胀反弹|通胀超预期|收紧",
    re.I,
)
_DOVISH_RE = re.compile(
    r"\b(?:rate cut|dovish|cooling inflation|disinflation|easing cycle|lower rates|softening labor|weak payrolls)\b|鸽派|降息|通胀降温|宽松",
    re.I,
)
_POS_TECH_RE = re.compile(
    r"\b(?:reinstat\w*|reappoint\w*|returns as ceo|unanimously reinstat\w*|beats|raised guidance|strong revenue|record quarter|buyback|ai demand surges|approved)\b|复职|重返.*ceo|超预期|上调指引|回购|获批",
    re.I,
)
_NEG_TECH_RE = re.compile(
    r"\b(?:resigns as ceo|steps down abruptly|accounting probe|sec investigation|subpoena|misses|cut guidance|antitrust suit|export curb|export ban|investigation|recall|downward revision|enforcement|hack|breach|outflow)\b|下调指引|反垄断|出口管制|调查|违规|制裁|外流|辞职",
    re.I,
)


def parse_aware_timestamp(raw: Any) -> datetime | None:
    """Parse ISO-8601 or RFC-2822 timestamp strictly requiring an explicit timezone."""
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip()
    try:
        iso_norm = text.replace("Z", "+00:00") if text.endswith("Z") else text
        dt = datetime.fromisoformat(iso_norm)
        if dt.tzinfo is not None and dt.utcoffset() is not None:
            return dt.astimezone(timezone.utc)
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(text)
        if dt.tzinfo is not None and dt.utcoffset() is not None:
            return dt.astimezone(timezone.utc)
    except (TypeError, ValueError, IndexError):
        pass
    return None


def _strip_tags(raw: str | None) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", raw or "")
    return html_lib.unescape(re.sub(r"\s+", " ", cleaned)).strip()


def _valid_http_url(url: Any) -> str | None:
    if not url or not isinstance(url, str):
        return None
    u = url.strip()
    try:
        parsed = urlparse(u)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    return u


def _tokens_for_dedup(title: str) -> set[str]:
    lowered = re.sub(r"\b(?:update\s*\d*|breaking|reuters|bloomberg|fed|press release|8-k|10-q)\b", " ", title.lower())
    return {t for t in re.findall(r"[a-z0-9]{2,}|[\u4e00-\u9fff]+", lowered) if len(t) >= 2}


def parse_rss_or_atom(xml_bytes: bytes, *, feed_id: str, feed_name: str, retrieved_at: datetime) -> list[dict[str, Any]]:
    """Parse official RSS 2.0 or Atom XML safely without resolving external entities."""
    if len(xml_bytes) > 2_000_000:
        raise ValueError(f"RSS feed too large: {feed_id}")
    raw_sha = hashlib.sha256(xml_bytes).hexdigest()
    text = xml_bytes.decode("utf-8", errors="replace").lstrip("\ufeff \r\n\t")
    if "<!ENTITY" in text.upper() or "<!DOCTYPE" in text.upper():
        # Strip DOCTYPE/ENTITY prologue conservatively before ElementTree parse
        text = re.sub(r"<!DOCTYPE[^>[]*(?:\[[^\]]*\])?\s*>", "", text, flags=re.I | re.S)
    root = ET.fromstring(text)
    items: list[dict[str, Any]] = []

    def _child_text(node: ET.Element, *names: str) -> str:
        for child in node:
            local = child.tag.rsplit("}", 1)[-1].lower()
            if local in names:
                if local == "link" and not (child.text or "").strip():
                    href = child.attrib.get("href")
                    if href:
                        return href.strip()
                if child.text and child.text.strip():
                    return child.text.strip()
        return ""

    # Handle RSS <item> and Atom <entry>
    entries = []
    for elem in root.iter():
        tag = elem.tag.rsplit("}", 1)[-1].lower()
        if tag in {"item", "entry"}:
            entries.append(elem)

    for entry in entries[:40]:
        title = _strip_tags(_child_text(entry, "title"))
        link = _child_text(entry, "link")
        pub_raw = _child_text(entry, "pubdate", "published", "updated", "date")
        summary = _strip_tags(_child_text(entry, "description", "summary", "content"))
        items.append(
            {
                "news_id": hashlib.sha256(f"{feed_id}|{link}|{title}".encode("utf-8")).hexdigest()[:16],
                "title": title,
                "url": link,
                "published_at_raw": pub_raw,
                "summary": summary[:600],
                "source_name": feed_name,
                "source_kind": "primary_official",
                "raw_sha256": raw_sha,
                "retrieved_at_utc": retrieved_at.astimezone(timezone.utc).isoformat(),
                "first_retrieved_at_utc": retrieved_at.astimezone(timezone.utc).isoformat(),
            }
        )
    return items


def fetch_official_news_feeds(
    *,
    timeout: float = 10.0,
    now: datetime | None = None,
    include_corporate_ir: bool = True,
    client: httpx.Client | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Fetch Fed / BLS / BEA & top NDX100 corporate IR Newsroom RSS/Atom releases (gracefully captures errors)."""
    retrieved = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    collected: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    own = client is None
    client = client or httpx.Client(timeout=timeout, follow_redirects=True, headers={"User-Agent": EDGAR_USER_AGENT})
    feeds = list(RSS_FEEDS) + (list(CORPORATE_IR_FEEDS) if include_corporate_ir else [])
    try:
        for fid, fname, url in feeds:
            try:
                host = urlparse(url).hostname
                if host not in ALLOWED_NEWS_HOSTS:
                    raise ValueError(f"disallowed host {host}")
                resp = client.get(url)
                final_host = (urlparse(str(resp.url)).hostname or "").lower()
                if final_host not in PRIMARY_OFFICIAL_HOSTS:
                    raise ValueError(f"redirected off allowlisted host {final_host}")
                resp.raise_for_status()
                collected.extend(parse_rss_or_atom(resp.content, feed_id=fid, feed_name=fname, retrieved_at=retrieved))
            except Exception as exc:
                errors[fid] = type(exc).__name__
    finally:
        if own:
            client.close()
    return collected, errors


def fetch_ndx_sec_edgar_filings(
    *,
    cutoff_utc: datetime,
    retrieved_at: datetime | None = None,
    lookback_hours: float = 72.0,
    timeout: float = 10.0,
    client: httpx.Client | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Fetch recent Form 8-K / 10-Q / 10-K filings for top NDX100 weights from official data.sec.gov."""
    cutoff_utc = cutoff_utc.astimezone(timezone.utc)
    ret_iso = (retrieved_at or cutoff_utc).astimezone(timezone.utc).isoformat()
    window_start = cutoff_utc - timedelta(hours=lookback_hours)
    filings_out: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    own = client is None
    client = client or httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": EDGAR_USER_AGENT, "Accept": "application/json"},
    )
    try:
        for ticker, cik in NDX_TOP_CIKS:
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            try:
                resp = client.get(url)
                resp.raise_for_status()
                raw = resp.content
                raw_sha = hashlib.sha256(raw).hexdigest()
                payload = resp.json()
                recent = ((payload.get("filings") or {}).get("recent") or {})
                forms = recent.get("form") or []
                acc_times = recent.get("acceptanceDateTime") or []
                acc_nums = recent.get("accessionNumber") or []
                primary_docs = recent.get("primaryDocument") or []
                items_col = recent.get("items") or []
                cik_int = str(int(cik))
                for idx, form in enumerate(forms[:30]):
                    if form not in {"8-K", "8-K/A", "10-Q", "10-K"}:
                        continue
                    raw_acc_dt = acc_times[idx] if idx < len(acc_times) else ""
                    pub_dt = parse_aware_timestamp(str(raw_acc_dt))
                    if pub_dt is None:
                        continue
                    if pub_dt > cutoff_utc or pub_dt < window_start:
                        continue
                    acc = str(acc_nums[idx]) if idx < len(acc_nums) else ""
                    doc = str(primary_docs[idx]) if idx < len(primary_docs) else ""
                    acc_nodash = acc.replace("-", "")
                    doc_url = (
                        f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{doc}"
                        if (acc_nodash and doc)
                        else f"https://www.sec.gov/edgar/browse/?CIK={cik}"
                    )
                    it_raw = items_col[idx] if idx < len(items_col) else ""
                    it_desc = ",".join(str(x) for x in it_raw) if isinstance(it_raw, list) else str(it_raw or "")
                    filings_out.append(
                        {
                            "news_id": f"edgar_{ticker}_{acc_nodash}"[:24],
                            "ticker": ticker,
                            "form": form,
                            "items": it_desc,
                            "title": f"SEC {form} Filing: {ticker} ({payload.get('name') or ticker}) items={it_desc or 'unspecified'}",
                            "summary": f"Official SEC EDGAR acceptance at {pub_dt.isoformat()}",
                            "url": doc_url,
                            "published_at_raw": pub_dt.isoformat(),
                            "source_name": "SEC EDGAR Official Submissions",
                            "source_kind": "primary_official",
                            "raw_sha256": raw_sha,
                            "retrieved_at_utc": ret_iso,
                            "first_retrieved_at_utc": ret_iso,
                        }
                    )
            except Exception as exc:
                errors[ticker] = type(exc).__name__
    finally:
        if own:
            client.close()
    return filings_out, errors


def _map_news_impact(item: dict[str, Any]) -> tuple[list[str], dict[str, str], str]:
    blob = f"{item.get('title', '')} {item.get('summary', '')}"
    explicit_syms = item.get("symbols") or item.get("affected_symbols")
    symbols: list[str] = []
    if isinstance(explicit_syms, list) and explicit_syms:
        symbols = [str(s) for s in explicit_syms if s in {"BTCUSDT", "XAUUSDT", "NDX100USDT"}]

    is_macro = bool(_MACRO_RE.search(blob))
    is_geo = bool(_GEO_RE.search(blob))
    is_btc = bool(_BTC_RE.search(blob))
    is_ndx = bool(_NDX_RE.search(blob) or item.get("ticker") in {"NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL"})
    is_gold = bool(_GOLD_RE.search(blob))

    if not symbols:
        if is_macro or is_geo:
            symbols = ["BTCUSDT", "XAUUSDT", "NDX100USDT"]
        else:
            if is_btc:
                symbols.append("BTCUSDT")
            if is_gold:
                symbols.append("XAUUSDT")
            if is_ndx:
                symbols.append("NDX100USDT")

    bias_map = {s: "neutral" for s in symbols}
    channel = "无直接匹配的跨市场价格传导"

    if is_macro:
        channel = "货币政策预期、贴现率与实际利率渠道"
        if _HAWKISH_RE.search(blob) and not _DOVISH_RE.search(blob):
            for s in symbols:
                bias_map[s] = "bearish"
        elif _DOVISH_RE.search(blob) and not _HAWKISH_RE.search(blob):
            for s in symbols:
                bias_map[s] = "bullish"
        else:
            for s in symbols:
                bias_map[s] = "two_sided_risk"
    elif is_geo:
        channel = "地缘/财政关税避险溢价与风险偏好分化渠道"
        for s in symbols:
            if s == "XAUUSDT":
                bias_map[s] = "bullish"
            elif s == "NDX100USDT":
                bias_map[s] = "bearish"
            else:
                bias_map[s] = "two_sided_risk"
    elif is_btc:
        channel = "现货 ETF 资金准入与加密监管合规渠道"
        if _POS_TECH_RE.search(blob) or re.search(r"\bnet inflow\b|\badoption\b|净流入", blob, re.I):
            bias_map["BTCUSDT"] = "bullish"
        elif _NEG_TECH_RE.search(blob):
            bias_map["BTCUSDT"] = "bearish"
        else:
            bias_map["BTCUSDT"] = "two_sided_risk"
    elif is_ndx:
        channel = "纳指核心权重股盈利预期与公司事件披露渠道"
        title_text = str(item.get("title") or "")
        if _POS_TECH_RE.search(title_text) and not _NEG_TECH_RE.search(title_text):
            bias_map["NDX100USDT"] = "bullish"
        elif _NEG_TECH_RE.search(title_text) and not _POS_TECH_RE.search(title_text):
            bias_map["NDX100USDT"] = "bearish"
        elif _POS_TECH_RE.search(blob) and not _NEG_TECH_RE.search(blob):
            bias_map["NDX100USDT"] = "bullish"
        elif _NEG_TECH_RE.search(blob) and not _POS_TECH_RE.search(blob):
            bias_map["NDX100USDT"] = "bearish"
        else:
            bias_map["NDX100USDT"] = "two_sided_risk"
    elif is_gold:
        channel = "央行储备/实物买盘与避险配置渠道"
        bias_map["XAUUSDT"] = "bullish" if re.search(r"\b(?:purchase|buy|reserve|add)\b|增持|购金", blob, re.I) else "two_sided_risk"

    # Allow caller-provided verified bias override in tests ONLY AFTER passing verification/time checks
    if isinstance(item.get("verified_bias_by_symbol"), dict):
        for s, b in item["verified_bias_by_symbol"].items():
            if s in bias_map and b in {"bullish", "bearish", "neutral", "two_sided_risk"}:
                bias_map[s] = b
    if isinstance(item.get("transmission_channel"), str) and item["transmission_channel"].strip():
        channel = item["transmission_channel"].strip()

    return symbols, bias_map, channel


def classify_and_filter_news_items(
    raw_items: list[dict[str, Any]],
    *,
    cutoff_utc: datetime,
    intraday_max_hours: float = 24.0,
    swing_max_hours: float = 72.0,
    duplicate_jaccard_min: float = 0.70,
    replay_mode: bool = False,
) -> list[dict[str, Any]]:
    """Verify timestamps/URLs, reject future/stale items, classify primary/secondary, and dedup reposts."""
    cutoff_utc = cutoff_utc.astimezone(timezone.utc)
    staged: list[dict[str, Any]] = []

    for idx, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title") or "").strip()
        url = _valid_http_url(raw.get("url"))
        pub_dt = parse_aware_timestamp(raw.get("published_at_utc") or raw.get("published_at_raw"))
        symbols, bias_map, channel = _map_news_impact(raw)
        if not symbols and raw.get("verified") is not False and pub_dt is not None and url is not None:
            # Skip irrelevant non-market notices (e.g., generic statistical methodology updates)
            continue

        first_ret = raw.get("first_retrieved_at_utc") or raw.get("retrieved_at_utc") or cutoff_utc.isoformat()
        nid = str(raw.get("news_id") or hashlib.sha256(f"{idx}|{url}|{title}".encode("utf-8")).hexdigest()[:16])
        if raw.get("verified") is False or (replay_mode and raw.get("live_scrape_without_pit_timestamp")) or not title or url is None or pub_dt is None:
            staged.append(
                {
                    "news_id": nid,
                    "title": title or "(untitled)",
                    "url": url,
                    "source_name": raw.get("source_name") or "unverified",
                    "raw_sha256": raw.get("raw_sha256"),
                    "published_at_utc": pub_dt.isoformat() if pub_dt else None,
                    "retrieved_at_utc": raw.get("retrieved_at_utc") or cutoff_utc.isoformat(),
                    "first_retrieved_at_utc": first_ret,
                    "verification_status": "unverified_unknown",
                    "freshness_status": "missing",
                    "news_kind": "unverified",
                    "duplicate_of": None,
                    "eligible_horizons": [],
                    "affected_symbols": symbols,
                    "bias_by_symbol": {s: "unknown" for s in symbols},
                    "transmission_channel": channel,
                    "limitation_note": "缺少可核验链接、明确时区发布时间或历史回放时点证明，按规则保留为未知（不计入方向）",
                }
            )
            continue

        age_hours = (cutoff_utc - pub_dt).total_seconds() / 3600.0
        if pub_dt > cutoff_utc:
            freshness = "future_excluded"
            horizons: list[str] = []
            note = "发布时间晚于当前决策截止时间（未来新闻，严格剔除）"
        elif age_hours > swing_max_hours:
            freshness = "stale_excluded"
            horizons = []
            note = f"超过 {swing_max_hours:g} 小时有效期（过期新闻，已剔除）"
        elif age_hours <= intraday_max_hours:
            freshness = "fresh"
            horizons = ["intraday", "swing_1_3d"]
            note = "日内与 1-3 日均有效的已核验消息"
        else:
            freshness = "lagged_usable"
            horizons = ["swing_1_3d"]
            note = "超过日内 24 小时但仍在 72 小时 1-3 日窗口内的已核验消息"

        host = (urlparse(url).hostname or "").lower()
        explicit_kind = str(raw.get("source_kind") or "")
        if _SECONDARY_RE.search(title) or explicit_kind in {"secondary", "secondary_interpretation", "commentary"}:
            nkind = "secondary_interpretation"
        elif host in PRIMARY_OFFICIAL_HOSTS or explicit_kind in {"primary", "primary_official", "primary_release"}:
            nkind = "primary_release"
        else:
            nkind = "secondary_interpretation"

        staged.append(
            {
                "news_id": nid,
                "title": title,
                "url": url,
                "source_name": raw.get("source_name") or host,
                "raw_sha256": raw.get("raw_sha256") or hashlib.sha256(f"{url}|{title}|{pub_dt.isoformat()}".encode("utf-8")).hexdigest(),
                "published_at_utc": pub_dt.isoformat(),
                "age_hours": round(age_hours, 2),
                "retrieved_at_utc": raw.get("retrieved_at_utc") or cutoff_utc.isoformat(),
                "first_retrieved_at_utc": first_ret,
                "verification_status": "verified",
                "freshness_status": freshness,
                "news_kind": nkind,
                "duplicate_of": None,
                "eligible_horizons": horizons,
                "affected_symbols": symbols,
                "bias_by_symbol": bias_map,
                "transmission_channel": channel,
                "limitation_note": note,
                "_tokens": _tokens_for_dedup(title),
                "_pub_dt": pub_dt,
            }
        )

    # Deduplicate valid items chronologically; keep earliest primary release as canonical source.
    active = [x for x in staged if x["verification_status"] == "verified" and x["freshness_status"] in {"fresh", "lagged_usable"}]
    active.sort(key=lambda x: (x["_pub_dt"], 0 if x["news_kind"] == "primary_release" else 1))
    kept: list[dict[str, Any]] = []
    for item in active:
        tok = item["_tokens"]
        dup_of = None
        if tok:
            for prev in kept:
                if not (set(item["affected_symbols"]) & set(prev["affected_symbols"])):
                    continue
                ptok = prev["_tokens"]
                if not ptok:
                    continue
                jaccard = len(tok & ptok) / len(tok | ptok)
                if jaccard >= duplicate_jaccard_min:
                    dup_of = prev["news_id"]
                    break
        if dup_of is not None:
            item["news_kind"] = "duplicate_repost"
            item["duplicate_of"] = dup_of
            item["limitation_note"] = f"与先发消息 `{dup_of}` 内容高度重复，判定为重复转载（零权重去重）"
        else:
            kept.append(item)

    for item in staged:
        item.pop("_tokens", None)
        item.pop("_pub_dt", None)
    return staged
