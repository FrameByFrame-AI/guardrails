#!/usr/bin/env python3
"""Generate synthetic Korean PII (opf span schema) to boost recall on the weak
classes: private_person, personal_handle, private_address.

Entities are embedded in varied casual/transactional Korean templates and are
deliberately followed by particles/honorifics (이/가/씨/야/에/에서/로/님/...) so the
model learns to recognize the entity even when the tokenizer glues a suffix onto its
last token. Output schema matches ko_pii_opf_v4:
  {"text": ..., "label": [{"category","start","end"}], "info": {"source":"synth"}}

Val/test are NOT touched — this only augments train.
"""
from __future__ import annotations
import argparse, json, random

SURNAMES = list("김이박최정강조윤장임한오서신권황안송류전홍고문양손배백허유남심노하곽성차주우구민")
GIVEN2 = ["민수","지영","승훈","수진","현우","은지","정민","하늘","소연","지훈","예진","준호","서윤","도현",
          "시우","유진","동현","지민","재현","미영","상우","보라","윤호","경아","태현","나영","성민","혜진",
          "진수","다은","예원","서준","하준","지호","주원","건우","민재","현준","우진","선우","서연","수아",
          "지우","채원","지윤","은서","유나","예은","수빈","소율","하영","민지","아린","다인","시윤","연우"]
GIVEN1 = ["철","훈","현","민","준","우","호","진","수","영","빈","결","찬","겸"]

ADJ = ["귀여운","행복한","외로운","용감한","조용한","빛나는","달리는","춤추는","웃는","꿈꾸는","날아가는","파란","붉은","작은"]
NOUN = ["고양이","호랑이","바다","하늘","별빛","구름","사자","여우","토끼","늑대","나비","장미","커피","달빛","바람"]
ENG_W = ["crimson","lemon","golden","silver","whisper","shadow","mango","violet","cosmic","frost","ember","willow",
         "ocean","maple","raven","lunar","stellar","pixel","velvet","amber","jade","onyx","coral","ivy","zest"]

SI = ["서울특별시","부산광역시","대구광역시","인천광역시","대전광역시","광주광역시","울산광역시","경기도","강원도",
      "충청북도","충청남도","전라북도","전라남도","경상북도","경상남도","제주특별자치도","세종특별자치시"]
# Standalone mid-size cities ending in '시' (a major MISSED bucket: 천안시/시흥시/파주시…)
CITY = ["천안시","시흥시","파주시","수원시","성남시","고양시","용인시","청주시","전주시","포항시","창원시","김해시",
        "의정부시","안산시","안양시","부천시","남양주시","화성시","평택시","김포시","광명시","군포시","이천시","양주시",
        "구리시","원주시","춘천시","강릉시","목포시","여수시","순천시","경주시","구미시","진주시","통영시","아산시",
        "서산시","당진시","제천시","충주시","익산시","군산시","경산시","양산시","거제시","세종시"]
GUN = ["평창군","양평군","가평군","홍천군","영월군","정선군","횡성군","무주군","장수군","함양군","고성군","청도군","의성군"]
MYEON = ["대화면","화천면","서면","북면","남면","동면","읍내면","청산면","오천면","가산면"]
RI = ["대화리","신촌리","상리","하리","중리","원리","대리","용산리","송정리","갈산리"]
GU = ["강남구","서초구","송파구","마포구","용산구","성동구","광진구","동대문구","종로구","중구","영등포구","관악구",
      "노원구","은평구","해운대구","수성구","유성구","분당구","일산동구","권선구"]
DONG = ["역삼동","신사동","청담동","삼성동","서초동","반포동","잠실동","구의동","효창동","상수동","가양동","목동",
        "정릉동","수당동","금오동","평창동","화곡동","개포동","대치동","방배동","부전동","행운동","연산동","대연동"]
RO = ["테헤란로","강남대로","세종대로","올림픽로","논현로","압구정로","양재대로","반포대로","천호대로","위례성대로"]


def rng(seed): return random.Random(seed)


def make_name(r):
    if r.random() < 0.85:
        return r.choice(SURNAMES) + r.choice(GIVEN2)
    return r.choice(SURNAMES) + r.choice(GIVEN1) + r.choice(GIVEN1)


def make_handle(r):
    k = r.random()
    if k < 0.4:
        return r.choice(ADJ) + r.choice(NOUN) + (str(r.randint(1, 999)) if r.random() < 0.4 else "")
    if k < 0.75:
        w = r.choice(ENG_W) + (r.choice(ENG_W) if r.random() < 0.4 else "")
        return w + str(r.randint(1, 9999))
    return r.choice(NOUN) + str(r.randint(1, 99))


def make_address(r):
    k = r.random()
    if k < 0.20:
        a = f"{r.choice(SI)} {r.choice(GU)} {r.choice(DONG)}"
    elif k < 0.34:
        a = f"{r.choice(GU)} {r.choice(DONG)}"
    elif k < 0.46:
        a = f"{r.choice(SI)} {r.choice(GU)} {r.choice(RO)} {r.randint(1,200)}"
    elif k < 0.58:
        a = r.choice(DONG)
    elif k < 0.72:                                   # standalone mid-size city '시'
        a = r.choice(CITY)
    elif k < 0.82:                                   # province + city
        a = f"{r.choice(SI)} {r.choice(CITY)}"
    elif k < 0.92:                                   # rural 군/면/리
        a = f"{r.choice(GUN)} {r.choice(MYEON)} {r.choice(RI)}"
    else:                                            # city + dong
        a = f"{r.choice(CITY)} {r.choice(DONG)}"
    if r.random() < 0.30:
        a += f" {r.choice(NOUN)}아파트 {r.randint(101,120)}동 {r.randint(101,2510)}호"
    return a


# Templates: list of (template_with_{e}, suffix_particle_pool). The entity is {e}.
PERSON_T = [
    "{e}{p} 어디 갔어?", "나 {e}{p} 잘 알아.", "{e}{p}한테 물어봐.", "어제 {e}{p}랑 밥 먹었어.",
    "{e}{p} 연락처 좀 알려줘.", "담당자 {e}{p} 맞으시죠?", "{e}{p} 자리에 없는데요.",
    "그 사람 이름이 {e}{p}야.", "저는 {e}{p}입니다.", "{e}{p} 보고 싶다.", "마케팅부 {e}{p} 내선 좀.",
    "{e}{p} 오늘 생일이래.", "우리 {e}{p} 진짜 착해.", "{e}{p}, 잠깐 이리 와봐.",
]
PERSON_P = ["", "이", "가", "씨", "야", "이가", "한테", "에게", "님", "이랑", "은", "는", "을", "를", "이는"]

HANDLE_T = [
    "내 닉네임은 {e}{p}.", "아이디 {e}{p}로 로그인했어.", "{e}{p}님 안녕하세요.", "게임에서 {e}{p}로 활동해.",
    "그 사람 계정이 {e}{p}야.", "{e}{p} 팔로우했어.", "닉 {e}{p} 어때?", "유저명 {e}{p}로 가입했습니다.",
    "{e}{p}이 댓글 달았더라.", "방금 {e}{p}한테 디엠 보냈어.", "핸들 {e}{p} 쓰고 있어.", "{e}{p} 채널 구독해.",
]
HANDLE_P = ["", "야", "이야", "님", "이", "가", "로", "이라고", "이에요", "이고", "한테", "는", "이랑"]

ADDR_T = [
    "주소는 {e}{p}.", "저 {e}{p} 살아요.", "{e}{p} 택배 보내주세요.", "집이 {e}{p}예요.",
    "{e}{p} 이사 왔어요.", "회사가 {e}{p} 있어요.", "{e}{p}로 와주세요.", "본가가 {e}{p}입니다.",
    "{e}{p} 근처에서 만나자.", "등록된 주소: {e}{p}", "{e}{p} 거주 중입니다.", "그분 댁이 {e}{p}래.",
]
ADDR_P = ["", "에", "에서", "이에요", "예요", "입니다", "으로", "쪽", "근처", "이", "고", "입니다."]


def assemble(template, entity, particle, category):
    """Fill {e}{p} keeping exact char offsets for the entity (NOT the particle)."""
    head, rest = template.split("{e}", 1)
    rest = rest.replace("{p}", particle, 1)
    text = head + entity + rest
    start = len(head)
    end = start + len(entity)
    return {"text": text, "label": [{"category": category, "start": start, "end": end}],
            "info": {"source": "synth"}}


def gen(category, n, seed):
    r = rng(seed)
    maker, temps, parts = {
        "private_person": (make_name, PERSON_T, PERSON_P),
        "personal_handle": (make_handle, HANDLE_T, HANDLE_P),
        "private_address": (make_address, ADDR_T, ADDR_P),
    }[category]
    out = []
    for _ in range(n):
        e = maker(r)
        rec = assemble(r.choice(temps), e, r.choice(parts), category)
        # sanity: offsets must reconstruct the entity
        s, en = rec["label"][0]["start"], rec["label"][0]["end"]
        assert rec["text"][s:en] == e, (rec["text"], e)
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-class", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    recs = []
    for i, cat in enumerate(["private_person", "personal_handle", "private_address"]):
        recs += gen(cat, args.per_class, args.seed + i)
    for rec in recs:  # offset self-check on every record
        sp = rec["label"][0]
        assert rec["text"][sp["start"]:sp["end"]], rec
    rng(args.seed).shuffle(recs)
    with open(args.out, "w", encoding="utf-8") as fh:
        for rec in recs:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(recs)} synthetic records -> {args.out}")
    from collections import Counter
    c = Counter(r["label"][0]["category"] for r in recs)
    print("per-class:", dict(c))


if __name__ == "__main__":
    main()
