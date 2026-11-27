#!/usr/bin/env python3
"""
构建 汉火识字 (Hanfire Shizi) 升级版古汉语语言模型 HXB2 (classical_bigram.bin.gz)：
  - 保留全量 ~11.5k 古籍词表 + ~350k 通用二元搭配（53.7万古诗文 + 39万成语词语 + 全部 Wave1-8 碑帖释文）
  - 新增 ~220k 高频三元搭配（3-gram）大幅提升整行/整碑长句 Viterbi 上下文纠偏
  - 新增 4 大文物专属分域先验（二元 + 三元）：
      0: oracle (殷墟甲骨卜辞套语)
      1: bronze (西周/东周青铜器铭文套语 + 利簋/颂鼎/何尊/毛公鼎/大盂鼎/散氏盘/虢季子白盘/墙盘/石鼓文/秦公簋等完整铭文)
      2: seal   (秦汉官印/私印/吉语印/瓦当套语)
      3: stele  (全部历代名碑名帖释文 + 战国楚简/睡虎地秦简/马王堆帛书高频套语)

紧凑排序二分查找格式 HXB2 (gzip 后约 2.2~2.8 MB，浏览器零哈希表开销)：
  Header (24B):
    [0..4]   magic = b"HXB2"
    [4..8]   n_vocab    (u32 LE)
    [8..12]  n_bigrams  (u32 LE)
    [12..16] n_trigrams (u32 LE)
    [16..20] n_dom_bi   (u32 LE)
    [20..24] n_dom_tri  (u32 LE)
  Body:
    vocab_cp : n_vocab * u32 LE           (严格升序)
    uni_q8   : n_vocab * u8               (0..255)
    bigrams  : n_bigrams * 5B             (<HHB: id1, id2, q8; 按 (id1<<16)|id2 升序)
    trigrams : n_trigrams * 7B            (<HHHB: id1, id2, id3, q8; 按 (id1, id2, id3) 升序)
    dom_bi   : n_dom_bi * 6B              (<BHHB: dom, id1, id2, q8; 按 (dom, id1, id2) 升序)
    dom_tri  : n_dom_tri * 8B             (<BHHHB: dom, id1, id2, id3, q8; 按 (dom, id1, id2, id3) 升序)
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sqlite3
import struct
import time
from collections import Counter
from pathlib import Path

try:
    from opencc import OpenCC
    _T2S = OpenCC("t2s")
except Exception:
    _T2S = None

SHIZI_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = Path("/mnt/s790/project/hanzi/hanzi-dict/hanzi.db")
DEFAULT_META = SHIZI_ROOT / "web/dist/assets/proto_meta.json.gz"
DEFAULT_OUT = SHIZI_ROOT / "web/dist/assets/classical_bigram.bin.gz"
MP_DIR = Path("/mnt/s790/project/hanzi_train/work_masterpieces")

CJK_SEG_RE = re.compile(r"[\u3400-\u9FFF]+")

# ---------- 4 大文物领域权威套语与经典铭文语料 ----------

# 0: 殷墟甲骨卜辞 (Oracle Bone Inscriptions)
ORACLE_CORPUS = [
    # 完整六十甲子 + 卜贞套语
    "甲子卜，乙丑卜，丙寅卜，丁卯卜，戊辰卜，己巳卜，庚午卜，辛未卜，壬申卜，癸酉卜。",
    "甲戌卜，乙亥卜，丙子卜，丁丑卜，戊寅卜，己卯卜，庚辰卜，辛巳卜，壬午卜，癸未卜。",
    "甲申卜，乙酉卜，丙戌卜，丁亥卜，戊子卜，己丑卜，庚寅卜，辛卯卜，壬辰卜，癸巳卜。",
    "甲午卜，乙未卜，丙申卜，丁酉卜，戊戌卜，己亥卜，庚子卜，辛丑卜，壬寅卜，癸卯卜。",
    "甲辰卜，乙巳卜，丙午卜，丁未卜，戊申卜，己酉卜，庚戌卜，辛亥卜，壬子卜，癸丑卜。",
    "甲寅卜，乙卯卜，丙辰卜，丁巳卜，戊午卜，己未卜，庚申卜，辛酉卜，壬戌卜，癸亥卜。",
    "癸卯卜，壳贞：旬亡祸。王占曰：有祟，其有来囏。乞至五日丁未，允有来囏自西。",
    "癸巳卜，壳贞：旬亡祸。王占曰：乃兹亦有祟，若偁。甲午王往逐兕，小臣叶车马。",
    "癸酉卜，宾贞：旬亡祸。王占曰：有祟，亡其来囏。癸未卜，争贞：旬亡祸。",
    "丙申卜，壳贞：来乙巳酒下乙。王占曰：酒，唯其有雨。乙巳夕有雨，允雨。",
    "贞：今夕亡祸。贞：今夕其雨。贞：今日不雨。贞：其雨。贞：不其雨。贞：允雨。",
    "贞：帝令雨足年。帝令雨弗其足年。贞：我受黍年。贞：受禾年。贞：受年。弗其受年。",
    "贞：王往田，亡灾。王其田，亡戈。王惠今日步，亡灾。王往于田，往来亡灾。",
    "贞：勿呼伐邛方，帝受我祐。贞：王伐土方，受有祐。贞：呼妇好先登人于庞。",
    "贞：妇好娩嘉。王占曰：其唯丁娩，嘉；其唯庚娩，弘吉。三旬又一日甲寅娩，不嘉，唯女。",
    "贞：王梦，亡祸。王疾齿，唯祖乙祟。贞：御于父乙。贞：告于大甲。贞：祷年于岳于河。",
    "贞：燎于河三牛，沈三牛。贞：岁于祖丁三牢。贞：侑于母庚。贞：翌乙亥其酒大乙。",
    "贞：大吉。弘吉。兹用。兹不用。上吉。小吉。吉。亡尤。亡灾。亡祸。亡害。亡咎。",
    "贞：出贞，永贞，亘贞，韦贞，旅贞，即贞，行贞，尹贞，大贞，兄贞，何贞，狄贞。",
    "在正月，在二月，在三月，在四月，在五月，在六月，在七月，在八月，在九月，在十月，在十一月，在十二月。",
    "王曰：其有来囏。允有来囏。自北，自东，自南，自西。不雨，大雨，小雨，延雨，启，大启。",
]

# 1: 西周/东周青铜器金文与石鼓文 (Bronze & Pre-Qin Stone Inscriptions)
BRONZE_CORPUS = [
    # 利簋
    "武王征商，唯甲子朝，岁鼎，克昏夙有商。辛未，王在阑师，赐有事利金，用作檀公宝尊彝。",
    # 何尊
    "唯王初迁宅于成周，复禀武王礼福自天。在四月丙戌，王诰宗小子于京室，曰：昔在尔考公氏，克逨文王，肆文王受兹大命。唯武王既克大邑商，则廷告于天，曰：余其宅兹中国，自兹乂民。呜呼！尔有唯小子亡识，视于公氏，有勋于天，彻命。敬享哉！唯王恭德裕天，训我不敏。王咸诰。何赐贝三十朋，用作庾公宝尊彝。唯王五祀。",
    # 颂鼎
    "唯三年五月既死霸甲戌，王在周康邵宫。旦，王格大室，即位。宰弘右颂入门，立中廷。尹氏受王命书，王呼史虢生册命颂。王曰：颂！命汝官司成周贮廿家，监司新造贮，用宫御。赐汝玄衣黹纯、赤市、朱黄、銮旂、攸勒，用事。颂拜稽首，受命册，佩以出，反入堇章。颂敢对扬天子丕显鲁休，用作朕皇考龚叔、皇母龚姒宝尊鼎，用追孝，蕲丐康虔、纯祐、通禄、永命。颂其万年眉寿，畯臣天子，灵终，子子孙孙永宝用。",
    # 虢季子白盘
    "唯十又二年正月初吉丁亥，虢季子白作宝盘。丕显子白，壮武于戎工，经维四方。搏伐玁狁，于洛之阳。折首五百，执讯五十，是以先行。桓桓子白，献馘于王。王孔加子白义，王格周庙，宣榭爰飨。王曰：白父，孔显有光。王赐乘马，是用佐王。赐用弓，彤矢其央。赐用钺，用政蛮方。子子孙孙，万年无疆。",
    # 毛公鼎
    "王若曰：父歆！丕显文武，皇天弘厌厥德，配我有周，膺受大命，率怀不廷方，亡不康静于文武耿光。唯天将集厥命，亦唯先正襄乂厥辟，劳勤大命，肆皇天亡斁，临保我有周，丕巩先王配命。旻天疾畏，嗣余小子弗及，邦将曷吉？呜呼！惧余小子家湛于艰，永巩先王。王曰：父歆！余唯肇经先王命，命汝乂我邦我家内外，憃于小大政，甈朕位，虩许上下若否。余非庸又昏，汝毋敢妄宁，虔夙夕，惠我一人，拥我邦小大猷，毋折缄，告余先王若德，用印邵皇天，緟恪大命，康能四国，俗我弗作先王忧。父歆！已曰及兹卿事寮、大史寮，于父即尹，命汝并司公族与参有司、小子、师氏、虎臣与朕亵事，以乃族干吾王身。赐汝秬鬯一卣、圭瓒宝、朱市、葱衡、玉环、玉瑑、金车、错衡、朱幩、虎幂、攸勒、马四匹、攸勒。歆拜稽首，敢对扬天子丕显休，用作尊鼎，子子孙孙永宝用。",
    # 大盂鼎
    "唯九月，王在宗周，命盂。王若曰：盂！丕显文王，受天有大命。在武王嗣文作邦，辟厥匿，匍有四方，畯正厥民。在雩御事，酗酒无敢醹，有柴烝祀无敢扰。故天翼临子，法保先王，匍有四方。我闻殷坠命，唯殷边侯甸与殷正百辟，率肆于酒，故丧师已。汝妹辰有大服，余唯即朕小学，汝勿克余乃辟一人。今我唯即型禀于文王正德，若文王命二三正。今余唯命汝盂，绍荣敬雍德经，敏朝夕入谏，享奔走，畏天畏。王曰：而！命汝盂型乃嗣祖南公。赐汝鬯一卣、冕衣、市、舄、车马，赐乃祖南公旂，用狩。赐汝邦司四伯，人鬲自驭至于庶人六百又五十又九夫；赐夷司王臣十又三伯，人鬲千又五十夫。盂用对王休，用作祖南公宝鼎，唯王廿又三祀。",
    # 史墙盘
    "曰古文王，初盭和于政，上帝降懿德大粤，匍有上下，会受万邦。武王既伐殷，微史烈祖乃来见武王，武王则命周公舍宇于周卑处。乙祖、亚祖、文考乙公，万年无疆，子子孙孙永宝用。",
    # 散氏盘
    "用夨扑散邑，乃即散用田。眉自滶涉，以南至于大沽，一封。以陟，二封，至于边柳。复涉滶，陟雩，鄾𨟹仆，以西，封于敝城楮木，封于刍逨，封于刍道。厥受图夨王于豆新宫东廷。厥左执要史正中农。",
    # 秦公簋
    "秦公曰：丕显朕皇祖，受天命，鼏宅禹迹，十又二公，在帝之坯。严恭寅天命，保业厥秦，虩事蛮夏。余虽小子，穆穆帅秉明德，烈烈桓桓，万民是敕。咸畜胤士，譪譪文武，镇静不廷，柔变百邦，于秦执事。作淑和钟，厥名曰哲邦，其音铣铣，雍雍孔煌，以邵格孝享，以受纯鲁多厘，眉寿无疆，畯惠在位，高弘有庆，匍佑四方，永宝宜。",
    # 石鼓文
    "我车既攻，我马既同。我车既好，我马既阜。君子员猎，员猎员游。麀鹿速速，君子之求。弯弯角弓，弓兹以寺。我驱其朴，其来趩趩。汧殹沔沔，烝彼淖渊。鰋鲤处之，君子渔之。漫漫有鲨，其游趣趣。帛鱼皪皪，其菹氐鲜。黄白其鯾，有鲂有鲌。其胡孔庶，脀之毚毚。田车孔安，鋚勒𨏟𨏟。六师既简，左骖幡幡，右骖騝騝。我以跻于原，我戎止陆。宫车其写，秀弓时射，麋豕孔庶。鹿鹿雉兔，其原有旜。",
    # 高频青铜器金文通用套语
    "唯王元年，唯王二年，唯王三年，唯王四年，唯王五年，唯王十又二年，唯王廿又三年。",
    "初吉甲子，初吉丁亥，初吉庚午，既生霸，既望，既死霸。王在周康宫，王在宗周，王在成周，王格大室，即位，立中廷。",
    "呼内史册命，赐玄衣黹纯、赤市、朱黄、銮旂、攸勒、秬鬯一卣、贝十朋、贝三十朋。",
    "拜稽首，敢对扬天子丕显休命，用作宝尊彝，用作宝尊鼎，用作旅簋，用作宝盘，用作旅彝。",
    "其万年眉寿无疆，用享孝于皇祖考，用祈眉寿，永命多福，子子孙孙永宝用享。",
]

# 2: 汉代玺印 / 官印 / 私印 / 吉语印 / 瓦当 (Han Seals & Tiles)
SEAL_CORPUS = [
    # 官职印与常用印尾
    "皇帝信玺，皇帝行玺，天子信玺，天子行玺，皇后之玺，丞相之印章，大司马印，大将军印章，车骑将军章，卫将军章，骠骑将军章，前将军印，后将军印，左将军印，右将军印。",
    "太尉之印，御史大夫章，大司农印，大司空印，大司徒印，光禄勋章，卫尉之印，太仆之印，廷尉之印，大鸿胪印，宗正之印，少府之印，执金吾印，将作大匠，水衡都尉。",
    "太守之印，太守章，内史之印，相印章，都尉之印，校尉之印，军司马印，别部司马，假司马印，军假司马，曲侯之印，军曲侯印，部曲将印，偏将军印章，裨将军印章。",
    "长史之印，主簿之印，功曹之印，祭酒之印，博士之印，郎中之印，侍中之印，中郎将印，虎贲中郎将，羽林中郎将，护军都尉，骑都尉印，奉车都尉，驸马都尉，关内侯印。",
    "亭侯之印，乡侯之印，列侯之印，县令之印，县长之印，县丞之印，县尉之印，市丞之印，仓丞之印，库丞之印，盐官之印，铁官之印，木官之印，水官之印，厨丞之印。",
    # 常见汉印姓氏 + 印尾组合
    "刘氏之印，张氏之印，王氏之印，李氏之印，陈氏之印，赵氏之印，杨氏之印，周氏之印，吴氏之印，徐氏之印，孙氏之印，马氏之印，朱氏之印，胡氏之印，郭氏之印，何氏之印，高氏之印，林氏之印，郑氏之印，梁氏之印，宋氏之印，谢氏之印，韩氏之印，唐氏之印，冯氏之印，曹氏之印，许氏之印，邓氏之印，萧氏之印，程氏之印，袁氏之印，田氏之印，董氏之印，潘氏之印，蔡氏之印，蒋氏之印，余氏之印，杜氏之印，叶氏之印，苏氏之印，魏氏之印，吕氏之印，丁氏之印，任氏之印，沈氏之印，姚氏之印，卢氏之印，傅氏之印，钟氏之印，姜氏之印，崔氏之印，谭氏之印，廖氏之印，范氏之印，汪氏之印，陆氏之印，金氏之印，石氏之印，戴氏之印，贾氏之印，韦氏之印，夏氏之印，侯氏之印，邹氏之印，熊氏之印，孟氏之印，秦氏之印，白氏之印，江氏之印，阎氏之印，薛氏之印，尹氏之印，段氏之印，雷氏之印，黎氏之印，史氏之印，陶氏之印，贺氏之印，顾氏之印，毛氏之印，郝氏之印，龚氏之印，邵氏之印，万氏之印，钱氏之印，严氏之印，武氏之印，孔氏之印，康氏之印。",
    "私印，之印，印信，印章，信印，私印信，白笺，言事，臣某，妾某，某之印，某私印，某印信，某印章。",
    # 汉代吉语印与瓦当文字
    "长乐未央，长生无极，长生未央，千秋万岁，延年益寿，大吉羊，大吉祥，宜子孙，宜官秩，宜官位，日利千万，日入千万，富贵昌，宜侯王，乐未央，毋相忘，长相思，常相思，出入大吉，敬事，正行，和众，思言，利出，大利，大吉，万岁，长乐，未央，汉并天下，与天无极，亿年无疆，永受嘉福，卫字瓦当，上林瓦当，宗正官当，都司空瓦。",
]

# 3: 简帛/碑帖高频经典语料补充 (Bamboo Slips + Classic Stele Addenda)
STELE_SLIP_EXTRA = [
    # 睡虎地秦简/楚简常见套语
    "田律，厩苑律，仓律，金布律，关市律，工律，工人程，均工，徭律，司空，军爵律，置吏律，效律，传食律，行书，内史杂，尉杂，属邦，法律答问，封诊式，为吏之道，日书。",
    "凡为吏之道，必精洁正直，慎谨坚固，审悉无私，微密纤察，安静毋苛，审当赏罚。",
    "凡良吏明法律令，事无不能也。有廉洁敦悫而好善，不喜为恶。君子之行，静以修身，俭以养德。",
    "昔者，凡人，君子曰，子曰，诗云，书曰，太一生水，水反辅太一，是以成天。天反辅太一，是以成地。天地复相辅也，是以成神明。神明复相辅也，是以成阴阳。",
    "道可道，非常道。名可名，非常名。无名天地之始，有名万物之母。上善若水，水善利万物而不争。",
    # 魏碑经典名句
    "唯大魏正光三年，唯大魏景明三年，唯大魏太和十九年，清信士，清信女，比丘，比丘尼，造石像一区，为皇帝陛下，为亡父母，为七世父母，所生父母，因缘眷属，愿托生西方妙乐国土，值佛闻法，一时成佛。",
    "君讳猛龙，字神回，南阳白水人也。其先出自轩辕，祖某，父某，器宇渊凝，风神秀拔，春秋若干，薨于位，乃作铭曰。",
]


def norm_text(s: str) -> str:
    if not s:
        return ""
    return _T2S.convert(s) if _T2S is not None else s


def feed_segments(text: str, valid_chars: set[str], uni: Counter, bi: Counter, tri: Counter, weight: int):
    for seg in CJK_SEG_RE.findall(text):
        p2 = p1 = None
        for ch in seg:
            if ch in valid_chars:
                uni[ch] += weight
                if p1 is not None:
                    bi[(p1, ch)] += weight
                    if p2 is not None:
                        tri[(p2, p1, ch)] += weight
                p2, p1 = p1, ch
            else:
                p2 = p1 = None


def score_ngrams_bi(items, uni: Counter):
    scored = []
    raw_scores = []
    for (c1, c2), cnt in items:
        p_cond = cnt / max(1.0, float(uni[c1]))
        p_rev = cnt / max(1.0, float(uni[c2]))
        sc = 0.55 * math.log1p(cnt) + 0.45 * math.log1p(math.sqrt(p_cond * p_rev) * 400.0)
        scored.append((c1, c2, sc))
        raw_scores.append(sc)
    return scored, (min(raw_scores) if raw_scores else 0.0), (max(raw_scores) if raw_scores else 1.0)


def score_ngrams_tri(items, bi: Counter, uni: Counter):
    scored = []
    raw_scores = []
    for (c1, c2, c3), cnt in items:
        denom_bi = max(1.0, float(bi.get((c1, c2), cnt)))
        denom_u = max(1.0, float(uni.get(c3, cnt)))
        p_cond = cnt / denom_bi
        p_rev = cnt / denom_u
        sc = 0.55 * math.log1p(cnt) + 0.45 * math.log1p(math.sqrt(p_cond * p_rev) * 300.0)
        scored.append((c1, c2, c3, sc))
        raw_scores.append(sc)
    return scored, (min(raw_scores) if raw_scores else 0.0), (max(raw_scores) if raw_scores else 1.0)


def main():
    ap = argparse.ArgumentParser(description="构建 HXB2 三元+四分域古汉语语言模型")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--proto-meta", type=Path, default=DEFAULT_META)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-bi", type=int, default=350000)
    ap.add_argument("--max-tri", type=int, default=220000)
    args = ap.parse_args()

    t0 = time.time()
    meta = json.loads(gzip.decompress(args.proto_meta.read_bytes()))
    valid_chars = {c for c in meta["chars"] if len(c) == 1 and 0x3400 <= ord(c) <= 0x9FFF}
    print(f"[HXB2] Loaded proto_meta CJK vocab: {len(valid_chars):,} chars", flush=True)

    uni = Counter()
    bi = Counter()
    tri = Counter()

    # 1) hanzi.db: 53.7万古诗文 + 39.2万成语词语
    con = sqlite3.connect(str(args.db))
    n_sent = 0
    for (sent,) in con.execute("SELECT sentence FROM sentences"):
        if not sent:
            continue
        n_sent += 1
        feed_segments(norm_text(sent), valid_chars, uni, bi, tri, weight=2)

    n_word = 0
    for word, wtype in con.execute("SELECT word, word_type FROM words"):
        if not word or wtype in ("字义", "谜语"):
            continue
        n_word += 1
        w = 3 if wtype == "成语" else 1
        for seg in CJK_SEG_RE.findall(norm_text(word)):
            if 2 <= len(seg) <= 10:
                feed_segments(seg, valid_chars, uni, bi, tri, weight=w)
    con.close()

    # 2) 全部 Wave1..Wave8 名碑名帖释文（叠加至通用 + stele 分域！）
    dom_uni = {d: Counter() for d in range(4)}
    dom_bi = {d: Counter() for d in range(4)}
    dom_tri = {d: Counter() for d in range(4)}

    seen_texts = set()
    n_stele_chars = 0
    for tf in (
        "transcripts.json", "transcripts_v3.json", "transcripts_v4.json",
        "transcripts_v5.json", "transcripts_v6.json", "transcripts_v7.json",
        "transcripts_v8.json",
    ):
        tp = MP_DIR / tf
        if not tp.exists():
            continue
        tdata = json.loads(tp.read_text(encoding="utf-8"))
        for _wk, info in tdata.items():
            raw_t = info.get("text", "") if isinstance(info, dict) else str(info)
            if not raw_t or raw_t[:60] in seen_texts:
                continue
            seen_texts.add(raw_t[:60])
            nt = norm_text(raw_t)
            n_stele_chars += len(nt)
            feed_segments(nt, valid_chars, uni, bi, tri, weight=8)
            feed_segments(nt, valid_chars, dom_uni[3], dom_bi[3], dom_tri[3], weight=4)

    # 3) 4 大文物专属分域语料（0=oracle, 1=bronze, 2=seal, 3=stele）
    for d_id, lines, gen_w, dom_w in [
        (0, ORACLE_CORPUS, 6, 12),
        (1, BRONZE_CORPUS, 6, 12),
        (2, SEAL_CORPUS, 6, 12),
        (3, STELE_SLIP_EXTRA, 6, 8),
    ]:
        for ln in lines:
            nt = norm_text(ln)
            feed_segments(nt, valid_chars, uni, bi, tri, weight=gen_w)
            feed_segments(nt, valid_chars, dom_uni[d_id], dom_bi[d_id], dom_tri[d_id], weight=dom_w)

    print(
        f"[HXB2] Scanned {n_sent:,} sents + {n_word:,} words + {len(seen_texts)} steles ({n_stele_chars:,} ch) "
        f"-> raw uni={len(uni):,}, bi={len(bi):,}, tri={len(tri):,} ({time.time()-t0:.1f}s)",
        flush=True,
    )

    # 词表（严格升序）
    vocab_chars = sorted(uni.keys(), key=lambda c: ord(c))
    ch2id = {c: i for i, c in enumerate(vocab_chars)}
    max_uni_log = math.log1p(max(uni.values()))
    uni_q8 = bytes(
        min(255, max(1, int(round(math.log1p(uni[c]) / max_uni_log * 255.0))))
        for c in vocab_chars
    )

    # 通用 Bigrams (>=3)
    kept_bi = [(k, c) for k, c in bi.items() if c >= 3]
    kept_bi.sort(key=lambda kv: kv[1], reverse=True)
    kept_bi = kept_bi[: args.max_bi]
    bi_scored, bi_min, bi_max = score_ngrams_bi(kept_bi, uni)
    bi_packed = []
    for c1, c2, sc in bi_scored:
        id1, id2 = ch2id[c1], ch2id[c2]
        q = min(255, max(1, int(round((sc - bi_min) / max(1e-6, bi_max - bi_min) * 255.0))))
        bi_packed.append(((id1 << 16) | id2, id1, id2, q))
    bi_packed.sort(key=lambda t: t[0])

    # 通用 Trigrams (>=4 过滤孤例，保留经典诗文/碑帖/成语三元组)
    kept_tri = [(k, c) for k, c in tri.items() if c >= 4]
    kept_tri.sort(key=lambda kv: kv[1], reverse=True)
    kept_tri = kept_tri[: args.max_tri]
    tri_scored, tri_min, tri_max = score_ngrams_tri(kept_tri, bi, uni)
    tri_packed = []
    for c1, c2, c3, sc in tri_scored:
        id1, id2, id3 = ch2id[c1], ch2id[c2], ch2id[c3]
        q = min(255, max(1, int(round((sc - tri_min) / max(1e-6, tri_max - tri_min) * 255.0))))
        tri_packed.append(((id1 << 32) | (id2 << 16) | id3, id1, id2, id3, q))
    tri_packed.sort(key=lambda t: t[0])

    # 分域 Bigrams & Trigrams
    dom_bi_packed = []
    dom_tri_packed = []
    for d_id in range(4):
        # stele 分域条目多，截断 Top 18k bi / 18k tri；oracle/bronze/seal 全保留
        lim = 18000 if d_id == 3 else 12000
        db_items = sorted(dom_bi[d_id].items(), key=lambda kv: kv[1], reverse=True)[:lim]
        dt_items = sorted(dom_tri[d_id].items(), key=lambda kv: kv[1], reverse=True)[:lim]
        db_sc, db_mn, db_mx = score_ngrams_bi(db_items, dom_uni[d_id])
        dt_sc, dt_mn, dt_mx = score_ngrams_tri(dt_items, dom_bi[d_id], dom_uni[d_id])
        for c1, c2, sc in db_sc:
            if c1 in ch2id and c2 in ch2id:
                id1, id2 = ch2id[c1], ch2id[c2]
                q = min(255, max(80, int(round((sc - db_mn) / max(1e-6, db_mx - db_mn) * 175.0 + 80.0))))
                dom_bi_packed.append(((d_id << 32) | (id1 << 16) | id2, d_id, id1, id2, q))
        for c1, c2, c3, sc in dt_sc:
            if c1 in ch2id and c2 in ch2id and c3 in ch2id:
                id1, id2, id3 = ch2id[c1], ch2id[c2], ch2id[c3]
                q = min(255, max(90, int(round((sc - dt_mn) / max(1e-6, dt_mx - dt_mn) * 165.0 + 90.0))))
                dom_tri_packed.append(((d_id << 48) | (id1 << 32) | (id2 << 16) | id3, d_id, id1, id2, id3, q))
        print(f"  Domain {d_id}: bi={len(db_items):,}, tri={len(dt_items):,}", flush=True)

    dom_bi_packed.sort(key=lambda t: t[0])
    dom_tri_packed.sort(key=lambda t: t[0])

    # 写出 HXB2 二进制 + gzip
    buf = bytearray()
    buf.extend(b"HXB2")
    buf.extend(struct.pack(
        "<IIIII",
        len(vocab_chars),
        len(bi_packed),
        len(tri_packed),
        len(dom_bi_packed),
        len(dom_tri_packed),
    ))
    for c in vocab_chars:
        buf.extend(struct.pack("<I", ord(c)))
    buf.extend(uni_q8)
    for _k, id1, id2, q in bi_packed:
        buf.extend(struct.pack("<HHB", id1, id2, q))
    for _k, id1, id2, id3, q in tri_packed:
        buf.extend(struct.pack("<HHHB", id1, id2, id3, q))
    for _k, d_id, id1, id2, q in dom_bi_packed:
        buf.extend(struct.pack("<BHHB", d_id, id1, id2, q))
    for _k, d_id, id1, id2, id3, q in dom_tri_packed:
        buf.extend(struct.pack("<BHHHB", d_id, id1, id2, id3, q))

    gz = gzip.compress(bytes(buf), compresslevel=6)
    args.out.write_bytes(gz)
    print(
        f"[HXB2] Wrote {args.out}: vocab={len(vocab_chars):,}, bi={len(bi_packed):,}, tri={len(tri_packed):,}, "
        f"dom_bi={len(dom_bi_packed):,}, dom_tri={len(dom_tri_packed):,} | raw={len(buf)/1e6:.2f}MB, gz={len(gz)/1e6:.2f}MB ({time.time()-t0:.1f}s)",
        flush=True,
    )

    # Sanity check 4 domains + classical trigrams
    tri_map = {(id1, id2, id3): q for _, id1, id2, id3, q in tri_packed}
    dtri_map = {(d, id1, id2, id3): q for _, d, id1, id2, id3, q in dom_tri_packed}
    for label, d_id, s in [
        ("oracle", 0, "癸卯卜"), ("oracle", 0, "旬亡祸"), ("oracle", 0, "王占曰"),
        ("bronze", 1, "永宝用"), ("bronze", 1, "拜稽首"), ("bronze", 1, "既死霸"),
        ("seal",   2, "太守章"), ("seal",   2, "长乐未"), ("seal",   2, "之印章"),
        ("stele",  3, "永和九"), ("stele",  3, "暮春之"), ("stele",  3, "修禊事"),
    ]:
        ids = tuple(ch2id.get(c, -1) for c in s)
        gq = tri_map.get(ids, 0)
        dq = dtri_map.get((d_id, *ids), 0)
        print(f"  [{label:6s}] {s}: gen_tri={gq:3d}/255 | dom_tri={dq:3d}/255")


if __name__ == "__main__":
    main()
