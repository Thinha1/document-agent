"""Generate a small synthetic TECHNICAL table set: spec tables in en/ja/vi, one question per table technique.

    python scripts/synth_tech.py build     # write examples/denso_tech_synthetic/ and the question set
    python scripts/synth_tech.py check     # verify every answer really is in its document
    python scripts/synth_tech.py ingest    # load the documents into a workspace (default denso-tech)

Why: the DENSO documents we can download are prose (policy, governance, reports);
their engineering documents are not public (denso.com refuses automated
downloads, and the one product manual that is public, the spark plug manual, has
charts and no tables). Retrieval over tables is a different problem from prose: a
row of numbers has no sentence around it, and once a table is cut into chunks the
later rows have lost their column headers. So this builds tables of the kind an
engineering department keeps, from a seeded random generator with NO model
involved: values are exact, every answer is checkable, the output is
byte-for-byte reproducible.

The question set is small on purpose: TEN facts, each one aimed at a specific way
a table breaks retrieval, asked in en, ja and vi (30 questions). `format` in the
question file names the technique, so the dashboard's per-format chips read as a
table-technique breakdown:

  grouped-header   the answer sits under a two-level header ("Grade 10.9 / Lubricated",
                   "Q3 / Appearance"); without the header the cell is just a number
  late-row         a row far down the table, in a chunk that no longer has the header
  chunk-boundary   a row the chunker cuts in half (chosen against the chunker as it
                   is when `build` runs; if the chunker is fixed the question stays
                   valid and simply stops being a boundary case)
  multi-row        the answer needs two rows (which supplier had the higher PPM)
  semantic-key     the row is named by its meaning in the question's language
                   ("the crankshaft sensor fault"), the document holds the code
  code-key         the row is named by a literal code (the easy control)
  plain-lookup     one cell, row and column named (the other control)

What is real and what is not: every value is fictional and every part number is
made up. The tables imitate the LAYOUT of torque charts, connector pin-outs,
fault-code lists, thermistor tables, sampling plans, supplier quality tables and
injector / ignition-coil data sheets. Nothing here is DENSO data, and each
document says so at the top. Scores compare retrieval configurations; they are not
a claim about DENSO's real documents.

Each document exists in ONE language, so for every fact one question is
same-language and two are cross-language over the same answer. Half of the
documents use Markdown pipe tables and half the aligned plain columns a PDF text
extractor produces (header once, at the top).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC_DIR = ROOT / "examples" / "denso_tech_synthetic"
QUESTIONS = ROOT / "evals" / "documents" / "synthetic_denso_tech.jsonl"
LANGS = ("en", "ja", "vi")
SEED = 20260928
CHUNK = (512, 120, 48)  # max, min, overlap tokens: the ingestion defaults

NOTICE = {
    "en": "SYNTHETIC EXAMPLE: fictional values generated for retrieval testing. Not DENSO data.",
    "ja": "合成例:検索評価用に生成した架空の値です。DENSO の実データではありません。",
    "vi": "VÍ DỤ TỔNG HỢP: giá trị hư cấu tạo ra để kiểm thử truy xuất. Không phải dữ liệu của DENSO.",
}

TOPICS = {
    "torque-spec": {"en": "the fastener tightening torque specification", "ja": "ボルト締付トルク仕様書", "vi": "tiêu chuẩn mô-men xiết bu lông"},
    "thermistor-table": {"en": "the coolant temperature sensor resistance table", "ja": "水温センサ抵抗表", "vi": "bảng điện trở cảm biến nhiệt độ nước"},
    "injector-flow": {"en": "the injector static flow data sheet", "ja": "インジェクタ静的流量データシート", "vi": "bảng dữ liệu lưu lượng tĩnh kim phun"},
    "connector-pinout": {"en": "the ECU connector pin assignment", "ja": "ECU コネクタのピン配列表", "vi": "bảng phân bố chân đầu nối ECU"},
    "sampling-plan": {"en": "the incoming inspection sampling plan", "ja": "受入検査の抜取表", "vi": "bảng lấy mẫu kiểm tra đầu vào"},
    "ignition-coil": {"en": "the ignition coil data sheet", "ja": "イグニッションコイル仕様表", "vi": "bảng dữ liệu bô-bin đánh lửa"},
    "fault-codes": {"en": "the diagnostic fault code list", "ja": "故障診断コード一覧", "vi": "danh mục mã lỗi chẩn đoán"},
    "supplier-ppm": {"en": "the quarterly supplier quality report", "ja": "四半期仕入先品質報告書", "vi": "báo cáo chất lượng nhà cung cấp theo quý"},
}


@dataclass
class Table:
    title: str
    intro: str
    header: list[list[str]]
    rows: list[list[str]]
    foot: str
    fact: Callable  # (frng, row index, mode) -> (keywords, {lang: question})
    pair: Callable | None = None  # (frng, row i, row j) -> the same, for a two-row question


def num(value: float, lang: str, digits: int = 1) -> str:
    text = f"{value:.{digits}f}"
    return text.replace(".", ",") if lang == "vi" else text


def render(header: list[list[str]], rows: list[list[str]], fmt: str) -> str:
    """Markdown pipe table, or the aligned columns of a text-extracted PDF."""
    width = max(len(r) for r in [*header, *rows])
    grid = [r + [""] * (width - len(r)) for r in [*header, *rows]]
    if fmt == "md":
        # header, separator, then any further header rows as ordinary rows: a valid table
        lines = ["| " + " | ".join(grid[0]) + " |", "|" + "|".join("---" for _ in range(width)) + "|"]
        lines.extend("| " + " | ".join(r) + " |" for r in grid[1:])
        return "\n".join(lines)
    widths = [max(len(r[i]) for r in grid) for i in range(width)]
    return "\n".join("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(r)).rstrip() for r in grid)


# --- the eight table types ----------------------------------------------------
# Each builder draws the TABLE from `rng` (so the documents never change when the
# questions do) and returns a dict: title, intro, header rows, body rows, footnote,
# and `fact(frng, i, mode)` -> (keywords, {lang: question}) for body row i.


def torque(rng, lang, topic):
    diameters = [5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 27, 30, 33, 36]
    sizes = [f"M{d}" for d in diameters]
    grades = ("8.8", "10.9", "12.9")
    rows, cells = [], {}
    for d, size in zip(diameters, sizes, strict=True):
        vals = {}
        for grade, k in zip(grades, (0.11, 0.155, 0.185), strict=True):
            dry = round(k * d**2.6 * rng.uniform(0.95, 1.05), 1)
            vals[(grade, "dry")] = dry
            vals[(grade, "lub")] = round(dry * rng.uniform(0.74, 0.8), 1)
        cells[size] = vals
        rows.append([size] + [num(vals[(g, c)], lang) for g in grades for c in ("dry", "lub")])
    head = [["Bolt", "Grade 8.8", "", "Grade 10.9", "", "Grade 12.9", ""], ["size"] + ["Dry (N·m)", "Lubricated (N·m)"] * 3]

    def fact(frng, i, mode=""):
        size, grade = sizes[i], frng.choice(["10.9", "12.9"])
        value = num(cells[size][(grade, "lub")], lang)
        return [size, value], {
            "en": f"What tightening torque in N·m does {topic['en']} give for an {size} bolt of grade {grade} in the lubricated condition?",
            "ja": f"{topic['ja']}では、強度区分 {grade} の {size} ボルトを潤滑状態で締め付けるときのトルクは何 N·m ですか？",
            "vi": f"Theo {topic['vi']}, bu lông {size} cấp bền {grade} ở trạng thái có bôi trơn cần xiết với mô-men bao nhiêu N·m?",
        }

    return Table(title="Fastener tightening torque", intro="Tightening torque by bolt size, strength grade and thread condition.",
                header=head, rows=rows, foot="Tolerance ±10 % of the nominal torque. Values apply to zinc-flake coated bolts.", fact=fact)


SIGNALS = [("CKP+", "クランク角信号(+)", "tín hiệu trục khuỷu (+)"), ("CKP-", "クランク角信号(-)", "tín hiệu trục khuỷu (-)"),
           ("CMP", "カム角信号", "tín hiệu trục cam"), ("THW", "水温センサ", "cảm biến nhiệt độ nước"),
           ("THA", "吸気温センサ", "cảm biến nhiệt độ khí nạp"), ("VG", "エアフロメータ", "lưu lượng khí nạp"),
           ("KNK", "ノックセンサ", "cảm biến kích nổ"), ("VTA", "スロットル開度", "độ mở bướm ga"),
           ("OX1", "酸素センサ", "cảm biến oxy"), ("IGT1", "点火信号 #1", "tín hiệu đánh lửa số 1"),
           ("IGT2", "点火信号 #2", "tín hiệu đánh lửa số 2"), ("INJ1", "噴射信号 #1", "tín hiệu phun số 1"),
           ("INJ2", "噴射信号 #2", "tín hiệu phun số 2"), ("ACC", "アクセサリ電源", "nguồn phụ kiện"),
           ("STA", "スタータ信号", "tín hiệu khởi động"), ("+B", "バッテリ電源", "nguồn ắc quy"),
           ("E01", "電源アース", "mass nguồn"), ("PRS", "燃圧センサ", "cảm biến áp suất nhiên liệu")]
WIRES = ["B", "W", "R", "G", "Y", "L", "BR", "GR", "LG", "P", "V", "O"]
RANGES = ["0.5–4.5", "0.3–4.7", "4.75–5.25", "9–16", "0.1–4.9", "2.4–2.6", "0.0–1.0", "1.8–3.2", "11–14.5", "0.8–4.2"]


def pinout(rng, lang, topic):
    tagged = [(f"B{bank}:{sig}", ja, vi) for bank in (1, 2, 3) for sig, ja, vi in SIGNALS]
    pins = list(range(1, len(tagged) + 1))
    rng.shuffle(pins)
    order = sorted(zip(pins, tagged, strict=True))
    rows = []
    for pin, (sig, ja, _vi) in order:
        wire = rng.choice(WIRES)
        rows.append([str(pin), sig, wire, ja, f"{rng.choice(RANGES)}"])
    head = [["ピン", "信号名", "線色", "機能", "電圧範囲 (V)"]]

    def fact(frng, i, mode=""):
        sig, vr = rows[i][1], rows[i][4]
        return [sig, vr], {
            "en": f"What voltage range in volts does {topic['en']} specify for the {sig} signal (bank {sig[1]})?",
            "ja": f"{topic['ja']}で、{sig} 信号(バンク {sig[1]})の電圧範囲は何 V と規定されていますか？",
            "vi": f"Theo {topic['vi']}, dải điện áp bao nhiêu V được quy định cho tín hiệu {sig} (dãy {sig[1]})?",
        }

    return Table(title="ECU コネクタのピン配列", intro="コネクタのピン配列と電圧範囲。", header=head, rows=rows,
                foot="線色: B=黒 W=白 R=赤 G=緑 Y=黄 L=青 BR=茶 GR=灰 LG=薄緑 P=桃 V=紫 O=橙", fact=fact)


DTC = [("P0335", "クランク位置センサ回路", "Mạch cảm biến vị trí trục khuỷu", "crankshaft position sensor circuit"),
       ("P0340", "カム位置センサ回路", "Mạch cảm biến vị trí trục cam", "camshaft position sensor circuit"),
       ("P0115", "水温センサ回路", "Mạch cảm biến nhiệt độ nước làm mát", "coolant temperature sensor circuit"),
       ("P0110", "吸気温センサ回路", "Mạch cảm biến nhiệt độ khí nạp", "intake air temperature sensor circuit"),
       ("P0100", "エアフロメータ回路", "Mạch cảm biến lưu lượng khí nạp", "mass airflow sensor circuit"),
       ("P0120", "スロットル位置センサ回路", "Mạch cảm biến vị trí bướm ga", "throttle position sensor circuit"),
       ("P0325", "ノックセンサ回路", "Mạch cảm biến kích nổ", "knock sensor circuit"),
       ("P0135", "酸素センサヒータ回路", "Mạch sấy cảm biến oxy", "oxygen sensor heater circuit"),
       ("P0190", "燃圧センサ回路", "Mạch cảm biến áp suất nhiên liệu", "fuel rail pressure sensor circuit"),
       ("P0500", "車速センサ", "Cảm biến tốc độ xe", "vehicle speed sensor")]


def dtc(rng, lang, topic):
    rows, info = [], {}
    for bank in (1, 2):
        for c, ja, vi, en in DTC:
            c = c if bank == 1 else c[:2] + "4" + c[3:]
            thr = f"> {num(rng.uniform(4.6, 4.95), lang, 2)} V / {rng.choice([1, 2, 3, 5])} s"
            mil = rng.choice(["ON", "ON", "OFF"])
            info[c] = (thr, f"{ja}(バンク{bank})", f"{vi} (dãy {bank})", f"{en} (bank {bank})")
            rows.append([c, f"{vi} (dãy {bank})", thr, mil, str(rng.choice([1, 2, 3]))])
    head = [["Mã lỗi", "Mô tả", "Điều kiện phát hiện", "Đèn MIL", "Số chu kỳ lái"]]

    def fact(frng, i, mode="code"):
        c = rows[i][0]
        thr, ja, vi, en = info[c]
        if mode == "code":
            q = {"en": f"What detection condition does {topic['en']} give for fault code {c}?",
                 "ja": f"{topic['ja']}では、故障コード {c} の検出条件はどう規定されていますか？",
                 "vi": f"Theo {topic['vi']}, điều kiện phát hiện của mã lỗi {c} là gì?"}
        else:
            q = {"en": f"What detection condition does {topic['en']} give for the {en} fault?",
                 "ja": f"{topic['ja']}では、{ja}の故障の検出条件はどう規定されていますか？",
                 "vi": f"Theo {topic['vi']}, điều kiện phát hiện của lỗi {vi.lower()} là gì?"}
        return [c, thr], q

    return Table(title="Danh mục mã lỗi chẩn đoán", intro="Mã lỗi, điều kiện phát hiện và trạng thái đèn báo lỗi.", header=head, rows=rows,
                foot="MIL: đèn báo lỗi động cơ. Số chu kỳ lái: số chu kỳ liên tiếp cần để bật đèn.", fact=fact)


def thermistor(rng, lang, topic):
    sensors = [(f"TS-{rng.randint(200, 899)}", rng.uniform(2.2, 2.6), rng.randint(3400, 3900)) for _ in range(3)]
    temps = list(range(-40, 151, 5))
    table = {}
    for pn, r25, b in sensors:
        for t in temps:
            table[(pn, t)] = r25 * math.exp(b * (1 / (t + 273.15) - 1 / 298.15))
    rows = [[str(t)] + [f"{table[(pn, t)]:.3f}" for pn, _, _ in sensors] for t in temps]
    head = [["Temp.", "Resistance (kΩ)", "", ""], ["(°C)"] + [pn for pn, _, _ in sensors]]

    def fact(frng, i, mode=""):
        t, pn = temps[i], frng.choice(sensors)[0]
        return [f"{table[(pn, t)]:.3f}"], {
            "en": f"What resistance in kΩ does {topic['en']} give for thermistor {pn} at {t} °C?",
            "ja": f"{topic['ja']}では、サーミスタ {pn} の {t} °C における抵抗値は何 kΩ ですか？",
            "vi": f"Theo {topic['vi']}, điện trở của nhiệt điện trở {pn} tại {t} °C là bao nhiêu kΩ?",
        }

    return Table(title="Thermistor resistance versus temperature", intro="Nominal resistance of three coolant temperature sensors.",
                header=head, rows=rows, foot="Tolerance ±2 % at 25 °C.", fact=fact)


def sampling(rng, lang, topic):
    lots = ["2–8", "9–15", "16–25", "26–50", "51–90", "91–150", "151–280", "281–500", "501–1200", "1201–3200",
            "3201–10000", "10001–35000", "35001–150000", "150001–500000"]
    n = [2, 3, 5, 8, 13, 20, 32, 50, 80, 125, 200, 315, 500, 800]
    aqls = ["0.25", "0.65", "1.0", "1.5", "2.5", "4.0"]
    rows, info = [], {}
    for lot, size in zip(lots, n, strict=True):
        cells = []
        for j, aql in enumerate(aqls):
            ac = max(0, int(size * (0.01 + 0.025 * j) * rng.uniform(0.6, 1.4)))
            cells.append(f"{ac}/{ac + 1}")
            info[(lot, aql)] = (size, f"{ac}/{ac + 1}")
        rows.append([lot, str(size)] + cells)
    head = [["ロットサイズ", "サンプル", "AQL", "", ""], ["(個)", "数 n"] + [f"{a} (Ac/Re)" for a in aqls]]

    def fact(frng, i, mode=""):
        lot, aql = lots[i], frng.choice(["1.5", "2.5", "4.0"])
        return [lot, info[(lot, aql)][1]], {
            "en": f"In {topic['en']}, what acceptance/rejection numbers apply to a lot of {lot} pieces at AQL {aql}?",
            "ja": f"{topic['ja']}で、ロットサイズ {lot} 個・AQL {aql} の合格判定数と不合格判定数はいくつですか？",
            "vi": f"Trong {topic['vi']}, lô {lot} chiếc ở mức AQL {aql} có số chấp nhận/bác bỏ là bao nhiêu?",
        }

    return Table(title="受入検査の抜取表", intro="ロットサイズ別のサンプル数と合否判定数。", header=head, rows=rows,
                foot="Ac: 合格判定個数、Re: 不合格判定個数。", fact=fact)


def supplier_ppm(rng, lang, topic):
    sups = sorted({f"S-{rng.randint(100, 899)}" for _ in range(44)})
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    data, rows = {}, []
    for s in sups:
        row = [s]
        for q in quarters:
            for cat in ("dim", "app"):
                data[(s, q, cat)] = rng.randint(12, 480)
                row.append(str(data[(s, q, cat)]))
        rows.append(row)
    head = [["Nhà cung cấp", "Q1", "", "Q2", "", "Q3", "", "Q4", ""], [""] + ["Kích thước", "Ngoại quan"] * 4]
    cat_names = {"dim": ("dimensional", "寸法", "kích thước"), "app": ("appearance", "外観", "ngoại quan")}

    def fact(frng, i, mode=""):
        s, q, cat = sups[i], frng.choice(quarters), "app"
        c_en, c_ja, c_vi = cat_names[cat]
        return [s, str(data[(s, q, cat)])], {
            "en": f"In {topic['en']}, what was the {c_en} defect rate in PPM of supplier {s} in {q}?",
            "ja": f"{topic['ja']}で、仕入先 {s} の {q} における{c_ja}不良率は何 PPM ですか？",
            "vi": f"Trong {topic['vi']}, tỷ lệ lỗi {c_vi} của nhà cung cấp {s} trong {q} là bao nhiêu PPM?",
        }

    def pair(frng, i, j):
        a, b, q, cat = sups[i], sups[j], frng.choice(quarters), "dim"
        c_en, c_ja, c_vi = cat_names[cat]
        return [a, str(data[(a, q, cat)]), b, str(data[(b, q, cat)])], {
            "en": f"In {topic['en']}, which of suppliers {a} and {b} had the higher {c_en} defect rate in {q}, and what were the two figures in PPM?",
            "ja": f"{topic['ja']}で、{q} の{c_ja}不良率が高かったのは仕入先 {a} と {b} のどちらですか？両者の値(PPM)も答えてください。",
            "vi": f"Trong {topic['vi']}, nhà cung cấp {a} hay {b} có tỷ lệ lỗi {c_vi} cao hơn trong {q}, và hai giá trị PPM là bao nhiêu?",
        }

    return Table(title="Chất lượng nhà cung cấp theo quý", intro="Tỷ lệ lỗi (PPM) theo nhà cung cấp và theo quý.", header=head, rows=rows,
                foot="PPM: số sản phẩm lỗi trên một triệu sản phẩm giao.", fact=fact, pair=pair)


def injector(rng, lang, topic):
    models = sorted({f"INJ-{rng.randint(300, 899)}" for _ in range(8)})
    pressures = list(range(200, 701, 25))
    data = {(m, p): round((rng.uniform(150, 420)) * math.sqrt(p / 300), 1) for m in models for p in pressures}
    rows = [[str(p)] + [f"{data[(m, p)]:.1f}" for m in models] for p in pressures]
    head = [["Fuel pressure", "Static flow (cc/min)"] + [""] * (len(models) - 1), ["(kPa)"] + models]

    def fact(frng, i, mode=""):
        p, m = pressures[i], frng.choice(models)
        return [f"{data[(m, p)]:.1f}"], {
            "en": f"What static flow in cc/min does {topic['en']} give for injector {m} at {p} kPa?",
            "ja": f"{topic['ja']}で、インジェクタ {m} の {p} kPa における静的流量は何 cc/min ですか？",
            "vi": f"Theo {topic['vi']}, lưu lượng tĩnh của kim phun {m} ở áp suất {p} kPa là bao nhiêu cc/min?",
        }

    return Table(title="Injector static flow by fuel pressure", intro="Static flow of eight injector models.", header=head, rows=rows,
                foot="Measured with n-heptane at 20 °C.", fact=fact)


def ignition_coil(rng, lang, topic):
    pns = sorted({f"IC-{rng.randint(2000, 9899)}" for _ in range(64)})
    data = {p: (round(rng.uniform(0.4, 0.9), 2), round(rng.uniform(6.5, 13.5), 1), round(rng.uniform(30, 45), 1), round(rng.uniform(2.0, 3.6), 1))
            for p in pns}
    rows = [[p, *(f"{v:.2f}" if i == 0 else f"{v:.1f}" for i, v in enumerate(data[p]))] for p in pns]
    head = [["品番", "一次抵抗", "二次抵抗", "出力電圧", "通電時間"], ["", "(Ω)", "(kΩ)", "(kV)", "(ms)"]]

    def fact(frng, i, mode=""):
        p = pns[i]
        return [p, f"{data[p][1]:.1f}"], {
            "en": f"What is the secondary resistance in kΩ of ignition coil {p} in {topic['en']}?",
            "ja": f"{topic['ja']}で、イグニッションコイル {p} の二次抵抗(kΩ)はいくつですか？",
            "vi": f"Trong {topic['vi']}, điện trở thứ cấp (kΩ) của bô-bin đánh lửa {p} là bao nhiêu?",
        }

    return Table(title="イグニッションコイル仕様", intro="品番ごとの一次・二次抵抗、出力電圧、通電時間。", header=head, rows=rows,
                foot="抵抗値は 20 °C での値。", fact=fact)


# (slug, doc language, format, builder, document code)
DOCS = [
    ("torque-spec", "en", "md", torque, "DT-TRQ-01"),
    ("thermistor-table", "en", "txt", thermistor, "DT-THM-02"),
    ("injector-flow", "en", "md", injector, "DT-INJ-03"),
    ("connector-pinout", "ja", "txt", pinout, "DT-PIN-04"),
    ("sampling-plan", "ja", "txt", sampling, "DT-AQL-05"),
    ("ignition-coil", "ja", "md", ignition_coil, "DT-IGC-06"),
    ("fault-codes", "vi", "md", dtc, "DT-DTC-07"),
    ("supplier-ppm", "vi", "txt", supplier_ppm, "DT-PPM-08"),
]

# The ten facts: (document, technique label, which row, mode). "late" = the last
# third of the table (a chunk without the header); "mid" = the first half.
PLAN = [
    ("torque-spec", "grouped-header", "late", ""),
    ("sampling-plan", "grouped-header", "mid", ""),
    ("supplier-ppm", "grouped-header", "late", ""),
    ("thermistor-table", "late-row", "late", ""),
    ("ignition-coil", "late-row", "late", ""),
    ("connector-pinout", "chunk-boundary", "boundary", ""),
    ("supplier-ppm", "multi-row", "pair", ""),
    ("fault-codes", "semantic-key", "late", "desc"),
    ("fault-codes", "code-key", "mid", "code"),
    ("injector-flow", "plain-lookup", "mid", ""),
]


def _row_spans(text: str, table: str, n_header_lines: int, n_rows: int) -> list[tuple[int, int]]:
    """(start, end) character offsets of each body row inside the document text."""
    offset = text.index(table)
    spans = []
    for k, line in enumerate(table.split("\n")):
        if k >= n_header_lines and len(spans) < n_rows:
            spans.append((offset, offset + len(line)))
        offset += len(line) + 1
    return spans


def boundary_row(text: str, spans: list[tuple[int, int]]) -> tuple[int, str]:
    """The body row the chunker cuts through (or, if it cuts between rows, the first
    row after the cut) at the first chunk boundary inside the table."""
    try:
        sys.path.insert(0, str(ROOT))
        from tini.documents.chunking import chunk_text

        chunks = chunk_text(text, *CHUNK)
    except ImportError:  # no dependencies: fall back to the last row
        return len(spans) - 1, "chunker unavailable"
    inside = [(a, b) for a, b in itertools.pairwise(chunks) if spans[0][0] < b.char_start <= spans[-1][1]]
    for a, b in inside:  # a row cut in half beats a clean cut
        for i, (start, end) in enumerate(spans):
            if start < b.char_start < end:
                return i, f"row cut in half by the chunker (chunk {a.ordinal}|{b.ordinal})"
    for a, b in inside:
        for i, (start, _end) in enumerate(spans):
            if start >= b.char_start:
                return i, f"first row after a cut (chunk {a.ordinal}|{b.ordinal}), header not repeated"
    return len(spans) - 1, "no chunk boundary inside the table"


def build_all() -> tuple[dict[str, str], list[dict]]:
    docs, cases, built = {}, [], {}
    for index, (slug, lang, fmt, builder, code) in enumerate(DOCS):
        rng = random.Random(SEED + index)
        t = builder(rng, lang, TOPICS[slug])
        table = render(t.header, t.rows, fmt)
        text = "\n".join([f"# {t.title} ({code})", "", NOTICE[lang], "", t.intro, "", table, "", t.foot, ""])
        name = f"tech-{slug}-{lang}"
        docs[f"{name}.{'md' if fmt == 'md' else 'txt'}"] = text
        n_header_lines = len(t.header) + (1 if fmt == "md" else 0)
        built[slug] = (name, fmt, t, _row_spans(text, table, n_header_lines, len(t.rows)), text)
    for k, (slug, technique, where, mode) in enumerate(PLAN):
        name, fmt, t, spans, text = built[slug]
        frng = random.Random(SEED * 7 + k)
        n = len(t.rows)
        note = ""
        if where == "pair":
            keywords, questions = t.pair(frng, frng.randrange(int(n * 0.1), int(n * 0.4)), frng.randrange(int(n * 0.7), n))
        else:
            if where == "boundary":
                i, note = boundary_row(text, spans)
            elif where == "late":
                i = frng.randrange(int(n * 0.7), n)
            else:
                i = frng.randrange(int(n * 0.15), int(n * 0.5))
            keywords, questions = t.fact(frng, i, mode)
        for q_lang in LANGS:
            case = {"id": f"{technique}-{slug}-{q_lang}", "lang": q_lang, "query": questions[q_lang], "expected_title": name,
                    "expected_keywords": keywords, "format": technique, "layout": "md-table" if fmt == "md" else "txt-table"}
            if note:
                case["note"] = note
            cases.append(case)
    return docs, cases


def check(docs: dict[str, str], cases: list[dict]) -> list[str]:
    problems = []
    by_title = {Path(n).stem: t for n, t in docs.items()}
    seen = {}
    for case in cases:
        text = by_title.get(case["expected_title"])
        if text is None:
            problems.append(f"{case['id']}: no document {case['expected_title']}")
            continue
        problems += [f"{case['id']}: keyword {kw!r} is not in the document" for kw in case["expected_keywords"] if kw not in text]
        if case["query"] in seen:
            problems.append(f"{case['id']}: same question as {seen[case['query']]}")
        seen[case["query"]] = case["id"]
        answer = case["expected_keywords"][-1]
        if answer in case["query"]:
            problems.append(f"{case['id']}: the answer {answer!r} appears in the question")
    per_fact = {}
    for case in cases:
        per_fact.setdefault(case["id"].rsplit("-", 1)[0], set()).add(case["lang"])
    problems += [f"{fact}: asked in {sorted(langs)}, expected {list(LANGS)}" for fact, langs in per_fact.items() if langs != set(LANGS)]
    if len(per_fact) != len(PLAN):
        problems.append(f"{len(per_fact)} facts, expected {len(PLAN)}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("build", "check", "ingest"))
    parser.add_argument("--workspace", default="denso-tech")
    args = parser.parse_args()
    docs, cases = build_all()
    if args.command == "build":
        DOC_DIR.mkdir(parents=True, exist_ok=True)
        for name, text in docs.items():
            (DOC_DIR / name).write_text(text, encoding="utf-8", newline="\n")
        QUESTIONS.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases), encoding="utf-8", newline="\n")
        print(f"wrote {len(docs)} documents to {DOC_DIR} and {len(cases)} questions to {QUESTIONS}")
    problems = check(docs, cases)
    if args.command in ("build", "check"):
        print("\n".join(problems) if problems else f"ok: {len(cases)} questions, every answer is in its document")
        return 1 if problems else 0
    if problems:
        print("\n".join(problems))
        return 1
    sys.path.insert(0, str(ROOT))
    from tini.config import load_settings
    from tini.db import connect
    from tini.documents.embeddings import make_embedding_provider
    from tini.documents.pipeline import IngestionPipeline
    from tini.documents.store import make_document_index

    settings = load_settings()
    conn = connect(settings.home)
    pipeline = IngestionPipeline(
        make_document_index(settings, conn, make_embedding_provider(settings)),
        table_chunking=settings.document_table_chunking,
    )
    for name in docs:
        print(pipeline.ingest_path(DOC_DIR / name, workspace_id=args.workspace))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
