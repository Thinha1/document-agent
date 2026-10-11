# General automotive technical documents (downloaded, not committed)

Public documents about the automotive industry in general (not DENSO), fetched with
plain requests for local retrieval testing. The PDFs are git-ignored; this file
records where each came from so they can be fetched again. Rights stay with their
publishers; nothing here is redistributed.

| file | language | what it is | source |
|---|---|---|---|
| `en-bosch-can-2.0-spec.pdf` | en | Bosch CAN Specification 2.0 (frame formats, bit timing, error handling), 73 pp | https://www.tekeye.uk/downloads/can2spec.pdf (a mirror of Bosch's freely distributed spec) |
| `en-autosar-cp-r22-11-sws-can-driver.pdf` | en | AUTOSAR Classic Platform R22-11, Specification of CAN Driver (API tables, configuration parameters), 133 pp | https://www.autosar.org/fileadmin/standards/R22-11/CP/AUTOSAR_SWS_CANDriver.pdf |
| `ja-mlit-saibet-068-rear-reflector.pdf` | ja | MLIT 保安基準の細目を定める告示 別添68 後部反射器の技術基準, 21 pp | https://www.mlit.go.jp/jidosha/kijyun/saimokubetten/saibet_068_00.pdf |
| `ja-mlit-ece-regulations-japanese-index.pdf` | ja | MLIT: UN/ECE regulations cited by the 細目告示, Japanese titles (1 page) | https://www.mlit.go.jp/jidosha/ece/eceindex.pdf |
| `vi-qcvn-09-2011-bgtvt-xe-o-to.pdf` | vi | QCVN 09:2011/BGTVT, vehicle technical safety and environmental regulation (superseded by 2015, then 2024), 31 pp | https://www.vr.org.vn/en-us/contents/Rules/Lists/Rules/Attachments/16/QCVN09.2011.BGTVT.pdf |
| `vi-qcvn-09-2024-du-thao-xe-o-to.pdf` | vi | QCVN 09:2024/BGTVT **draft** (dự thảo), 96 pp | https://vibonline.com.vn/wp-content/uploads/2024/01/03_DT_QCVN09_2024_BGTVT-31.05.2024.pdf |

Not fetched, on purpose: UN/ECE regulations (R13-H braking, R100 electric power
train) on `unece.org` answer automated clients with 403, and EUR-Lex's copies with
an asynchronous bot check (HTTP 202). Both are refusals to automated access, so
they were not worked around; download them in a browser if you want them. The
NXP-hosted copy of the CAN spec and two Vietnam Register URLs returned 404.

Note the two QCVN 09 files: the 2011 text is a superseded edition and the 2024 one
is a draft, which makes them a natural test for the catalog's `superseded` status.
