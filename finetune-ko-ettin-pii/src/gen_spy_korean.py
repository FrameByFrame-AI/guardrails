#!/usr/bin/env python3
"""Path B: native Korean multi-domain PII generation (SPY methodology).

Generate realistic Korean documents with PII placeholders via a local LLM, then fill
placeholders with Faker(ko_KR) + Korean RRN/date/account, recording exact char spans.
Runs on the HOST (talks to the vLLM endpoint at localhost; uses Faker on CPU).
"""
import argparse, json, random, re, sys, time
import requests
from faker import Faker

ENDPOINT = "http://localhost:47821/v1/chat/completions"
MODEL = "gemma-4-31B-it-NVFP4"
fk = Faker("ko_KR")
rng = random.Random(42)

PH_MAP = {  # placeholder -> our class
    "PERSON": "private_person", "ADDRESS": "private_address", "PHONE": "private_phone",
    "EMAIL": "private_email", "DATE": "private_date", "URL": "private_url",
    "ACCOUNT": "account_number", "HANDLE": "personal_handle", "IP": "ip_address",
}
DOMAINS = {
    "의료": ["진료 상담", "처방 안내", "건강검진 결과", "예약 문자"],
    "법률": ["법률 상담 질문", "계약서 조항", "진술서", "내용증명"],
    "금융": ["대출 신청서", "계좌 개설", "거래 내역 문의", "카드 분실 신고"],
    "고객지원": ["문의 티켓", "불만 접수", "환불 요청", "챗봇 대화"],
    "이커머스": ["주문 확인", "배송 안내", "반품 신청"],
    "보험": ["보험금 청구", "가입 상담"],
    "인사": ["이력서", "인사 기록", "재직 증명 요청"],
    "부동산": ["임대차 계약", "매물 문의"],
    "행정": ["민원 신청", "증명서 발급 요청"],
    "이메일": ["업무 이메일", "안내 메일"],
    # handle-rich domains (boost personal_handle)
    "소셜미디어": ["SNS 게시물", "프로필 소개", "댓글/멘션", "팔로우 안내", "DM 대화"],
    "게임": ["게임 계정 안내", "길드원 모집", "랭킹 공지", "인게임 채팅", "아이템 거래"],
    "커뮤니티": ["포럼 게시글", "회원 가입 안내", "쪽지", "닉네임 변경 요청"],
    "IT지원": ["계정/로그인 문제", "비밀번호 재설정", "아이디 찾기", "오류 신고"],
    "통신": ["통신 가입 신청", "요금 문의", "번호 이동"],
    "교육": ["수강 신청", "성적 안내", "학사 문의"],
    "여행": ["예약 확인", "항공권 발권", "숙소 예약"],
    "배달": ["배달 주문", "배달 문의", "리뷰 작성"],
}


def _phone():
    return rng.choice([f"010-{rng.randint(1000,9999)}-{rng.randint(1000,9999)}",
                       f"02-{rng.randint(200,999)}-{rng.randint(1000,9999)}",
                       f"0{rng.randint(31,64)}-{rng.randint(200,999)}-{rng.randint(1000,9999)}"])
def _rrn():
    return f"{rng.randint(0,99):02d}{rng.randint(1,12):02d}{rng.randint(1,28):02d}-{rng.randint(1,4)}{rng.randint(100000,999999)}"
def _account():
    return rng.choice([_rrn(), f"{rng.randint(100,999)}-{rng.randint(10,99)}-{rng.randint(100000,999999)}",
                       f"{rng.randint(1000,9999)}-{rng.randint(1000,9999)}-{rng.randint(1000,9999)}-{rng.randint(1000,9999)}"])
def _date():
    return rng.choice([f"{rng.randint(1960,2024)}년 {rng.randint(1,12)}월 {rng.randint(1,28)}일",
                       f"{rng.randint(70,99)}년 {rng.randint(1,12)}월 {rng.randint(1,28)}일",
                       f"{rng.randint(1980,2024)}.{rng.randint(1,12):02d}.{rng.randint(1,28):02d}"])
# realistic Korean handle distribution (~66% hangul nicknames / ~31% ascii word+digits)
_NA = ["미국","행복","빛나는","날아","꿈꾸는","귀여운","외로운","조용한","파란","작은","하얀","검은","웃는","시리우스","제이켄"]
_NB = ["마미","엄마","고양이","호랑이","바다","하늘","별","구름","천사","왕자","공주","대장","박사","요정","곰돌이","토끼","케이","한라","재롱이"]
_EW = ["crimson","lemon","golden","silver","shadow","mango","violet","frost","ocean","maple","raven","lunar","pixel","amber","jade","rainbow","dune","thunder","forest","mink"]
def _handle():
    k = rng.random()
    if k < 0.33:
        w = rng.choice(_EW) + (rng.choice(_EW) if rng.random() < 0.4 else "")
        return w + (str(rng.randint(1, 999999))[:rng.randint(1, 6)] if rng.random() < 0.85 else "")
    base = rng.choice(_NA) + rng.choice(_NB) if rng.random() < 0.6 else rng.choice(_NB) + (rng.choice(_NB) if rng.random() < 0.3 else "")
    if rng.random() < 0.25: base += str(rng.randint(1, 999))
    if rng.random() < 0.2 and not base.endswith("이"): base += "이"
    return base
def gen_entity(t):
    if t == "PERSON": return fk.name()
    if t == "ADDRESS": return fk.address().replace("\n", " ")
    if t == "PHONE": return _phone()
    if t == "EMAIL": return fk.ascii_email()
    if t == "DATE": return _date()
    if t == "URL": return fk.uri()
    if t == "ACCOUNT": return _account()
    if t == "HANDLE": return _handle()
    if t == "IP": return fk.ipv4()
    return ""

PROMPT = """당신은 한국어 합성 데이터 생성기입니다. '{domain}' 분야의 현실적인 한국어 문서 {k}개를 생성하세요.
문서 유형 예시: {types}.
규칙:
- 모든 개인식별정보(PII)는 반드시 다음 자리표시자로만 표기하세요: <PERSON> <ADDRESS> <PHONE> <EMAIL> <DATE> <URL> <ACCOUNT> <HANDLE> <IP>
- 사용자명·아이디·닉네임·계정명은 반드시 <HANDLE>로 표기하세요(가능하면 자주 포함).
- 실제 이름·번호·주소를 쓰지 말고 위 자리표시자만 사용하세요.
- 문서마다 길이·형식·포함되는 PII 종류를 다양하게 하세요. 한 문서에 PII 2~5개 포함.
- 자연스럽고 현실적인 한국어로 작성하세요.
- 출력은 JSON 문자열 배열만. 설명·코드블록 금지. 예: ["문서1", "문서2"]"""

PH_RE = re.compile(r"<([A-Z]+)>")


def fill(doc):
    out, labels, pos = "", [], 0
    for m in PH_RE.finditer(doc):
        t = m.group(1)
        if t not in PH_MAP:
            return None
        out += doc[pos:m.start()]
        val = gen_entity(t)
        s = len(out); out += val; e = len(out)
        labels.append({"category": PH_MAP[t], "start": s, "end": e})
        pos = m.end()
    out += doc[pos:]
    if not labels:
        return None
    return {"text": out, "label": labels, "info": {"source": "spy_ko"}}


def call(domain, types, k, temp):
    body = {"model": MODEL, "temperature": temp, "max_tokens": 2000,
            "messages": [{"role": "user", "content": PROMPT.format(domain=domain, types=", ".join(types), k=k)}]}
    r = requests.post(ENDPOINT, json=body, timeout=180)
    txt = r.json()["choices"][0]["message"]["content"].strip()
    txt = re.sub(r"^```(json)?|```$", "", txt, flags=re.M).strip()
    try:
        docs = json.loads(txt)
        return [d for d in docs if isinstance(d, str)]
    except Exception:
        return []


def _one_call(task):
    dom, types, per_call = task
    try:
        docs = call(dom, types, per_call, temp=rng.uniform(0.7, 1.0))
    except Exception:
        return dom, []
    out = []
    for d in docs:
        rec = fill(d)
        if rec:
            rec["info"]["domain"] = dom
            out.append(rec)
    return dom, out


def main():
    from concurrent.futures import ThreadPoolExecutor
    from collections import Counter
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--docs-per-domain", type=int, default=600)
    ap.add_argument("--per-call", type=int, default=5)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--domains", default="")
    args = ap.parse_args()
    domains = args.domains.split(",") if args.domains else list(DOMAINS)
    calls_per_dom = (args.docs_per_domain + args.per_call - 1) // args.per_call
    tasks = [(d, DOMAINS[d], args.per_call) for d in domains for _ in range(calls_per_dom + 2)]  # headroom
    per = Counter(); n = 0; t0 = time.time()
    with open(args.out, "w", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=args.workers) as ex:
        for dom, recs in ex.map(_one_call, tasks):
            for rec in recs:
                if per[dom] >= args.docs_per_domain:
                    break
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); per[dom] += 1; n += 1
            if n and n % 200 == 0:
                print(f"  {n} docs ({time.time()-t0:.0f}s) {dict(per)}", flush=True)
    print(f"wrote {n} docs -> {args.out}")
    print("per-domain:", dict(per))


if __name__ == "__main__":
    main()
