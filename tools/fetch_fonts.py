"""Download the open Arabic typefaces the generator was built and tested with.

36 files from Google Fonts, all under the SIL Open Font License: Naskh, Kufi,
Ruqaa, Nastaliq and Maghribi faces, a few modern sans, and several with real
bold weights or a weight axis. Pinned to one commit of github.com/google/fonts
and checked against SHA-256, and saved under the names the generator expects:
`Family.ttf`, and `Family-Bold.ttf` beside it, which is how a bold face is
found. Each family's licence is saved under LICENSES/.

    python tools/fetch_fonts.py fonts/
    synthetic-pages --fonts fonts/ ...

Any other Arabic .ttf or .otf works too; these are what the defaults were
tuned on.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

COMMIT = "6c1dbe27dcf7d817d4b62cec033ba77fa8e2a5ec"
RAW = f"https://raw.githubusercontent.com/google/fonts/{COMMIT}/"

#: saved as -> (path in google/fonts, sha256)
FONTS = {
    "Alkalami.ttf": ("ofl/alkalami/Alkalami-Regular.ttf", "4410dfc591b918dd4360b0252b079855a8a00250de363c8c0163b29ddfc56434"),
    "Almarai-Bold.ttf": ("ofl/almarai/Almarai-Bold.ttf", "06401391b5b39fbcf2bc0c6a6e03faf992435e7b011183a1861737e913af8613"),
    "Almarai.ttf": ("ofl/almarai/Almarai-Regular.ttf", "5b5cad33c8e5922083728d271cdcda91f96ba291f1307eccc4f4ef732dc19622"),
    "Amiri-Bold.ttf": ("ofl/amiri/Amiri-Bold.ttf", "cfccb794268e7d573d857e6d6a67f89cf8a053e8ffd85dfa0c8ec1bb36fc4827"),
    "Amiri.ttf": ("ofl/amiri/Amiri-Regular.ttf", "ab391c4147d054c48976e98322ad0eefe1427aa0e0502a12a4c75d80a70cfcd7"),
    "ArefRuqaa-Bold.ttf": ("ofl/arefruqaa/ArefRuqaa-Bold.ttf", "247071015b7eefd63f94d6e47949c5d10294ed31bd432f809ba9a219d93f91bb"),
    "ArefRuqaa.ttf": ("ofl/arefruqaa/ArefRuqaa-Regular.ttf", "ceb786d83ba92f35e96efcd8623c2858d288e0c543c8761ba35b1020989464f9"),
    "ArefRuqaaInk-Bold.ttf": ("ofl/arefruqaaink/ArefRuqaaInk-Bold.ttf", "601b5c607e7e20e7957673b92947d35eb87620cd406214037a140695221c31b4"),
    "ArefRuqaaInk.ttf": ("ofl/arefruqaaink/ArefRuqaaInk-Regular.ttf", "0415c2410b1e453f5d0ea0db114fb95d01f65ba2e9e2dc4ae67c5fb73c33e863"),
    "Cairo.ttf": ("ofl/cairo/Cairo[slnt,wght].ttf", "667c987182391c91f4e57a2f455b1794fb5e3ee6ca4ef3383e86bb690fa9c964"),
    "Changa.ttf": ("ofl/changa/Changa[wght].ttf", "7c4f7a14d4b70ac8816ea8df3a0b127aee4c5f5af7239aca2afecc84ddc7f4d3"),
    "ElMessiri.ttf": ("ofl/elmessiri/ElMessiri[wght].ttf", "5d58d5259d2accccba39cffe12e1226e88b6afc5150e3605eb2a9c2fe06ccab0"),
    "Gulzar.ttf": ("ofl/gulzar/Gulzar-Regular.ttf", "246337973ac64e8b70fe611405dcf32a395a8885c4497dc45dbc4975d3ac0831"),
    "Harmattan-Bold.ttf": ("ofl/harmattan/Harmattan-Bold.ttf", "eb9de49ecb72d34e14c768bea6f9a7831f6bd2c15b062fcc2bb17b0ac967bc67"),
    "Harmattan.ttf": ("ofl/harmattan/Harmattan-Regular.ttf", "5fbaafc51ad21663729b168afaff5f28d3e3087f1fc736ada32d7cd99674c0e0"),
    "IBMPlexSansArabic-Bold.ttf": ("ofl/ibmplexsansarabic/IBMPlexSansArabic-Bold.ttf", "691e0c891a38637ae6bbdb69700f8042cb0724a137bee615068ffdb92244f61f"),
    "IBMPlexSansArabic.ttf": ("ofl/ibmplexsansarabic/IBMPlexSansArabic-Regular.ttf", "6f611412270a132bbac838da9259d4c68569b4175f3b3b8fa3fa36a30b56dab9"),
    "Kufam.ttf": ("ofl/kufam/Kufam-Italic[wght].ttf", "37c5cf29cb515099783461051cf2c927fc18605b3e7e0088f9bd2150aed756a9"),
    "Lateef-Bold.ttf": ("ofl/lateef/Lateef-Bold.ttf", "eee405eeb4e81491b0b659fb40dc016765292022d74406c616f72501e160afe2"),
    "Lateef.ttf": ("ofl/lateef/Lateef-Regular.ttf", "fcbd9e0227a4d7238880defa5fd74c012d69351150709e559a8577446145c2fc"),
    "Mada.ttf": ("ofl/mada/Mada[wght].ttf", "bf4d193ee78cea2fd8bdc9e64e5cb7e8a19ff76ab74f97a9a4a0e80f8f4cefde"),
    "MarkaziText.ttf": ("ofl/markazitext/MarkaziText[wght].ttf", "424b7331681160ff2a4b209bfc356ea9564d429a7daea43ee4dc28de8ff4f9fe"),
    "NotoKufiArabic.ttf": ("ofl/notokufiarabic/NotoKufiArabic[wght].ttf", "494f6b61469d7a02a2d63f0fc4930bb007388d8cfe551de5eb98354e100889f3"),
    "NotoNaskhArabic.ttf": ("ofl/notonaskharabic/NotoNaskhArabic[wght].ttf", "67b5a525a661b607971fbd3f96a81b89d3a768e74534fca84f18ac97e6fab72f"),
    "NotoNastaliqUrdu.ttf": ("ofl/notonastaliqurdu/NotoNastaliqUrdu[wght].ttf", "98a4787f34eb6fde57fb9a1121a8f216301196ab0da98ea110cad581be8abbcd"),
    "NotoSansArabic.ttf": ("ofl/notosansarabic/NotoSansArabic[wdth,wght].ttf", "63111b5b2e074dd48cc67692e0a2726d86ee94c1c37fe8598257b7b4e87e869e"),
    "Qahiri.ttf": ("ofl/qahiri/Qahiri-Regular.ttf", "9c4d8cfba7b663dad7eeeeb52a625749f3e1f5ec472a0a6333e1684a2f43dbdf"),
    "ReadexPro.ttf": ("ofl/readexpro/ReadexPro[HEXP,wght].ttf", "268bba7e1e8f3b14d798b3fb0e40ebaa3fc39308c9ac0020e2faf6df181cc30e"),
    "ReemKufi.ttf": ("ofl/reemkufi/ReemKufi[wght].ttf", "33ead1986cd48c138c4d906c9e5a6341877268461f97623409a2014de1cd4aa4"),
    "ReemKufiFun.ttf": ("ofl/reemkufifun/ReemKufiFun[wght].ttf", "d1342cebd08afc04a51bf9a28e5ac6a8299c5716c47df9b7ea27f94653aaa40f"),
    "ReemKufiInk.ttf": ("ofl/reemkufiink/ReemKufiInk-Regular.ttf", "45fecf59cff786a23357b61d282e5ad383438be9507f3aa9ea00eb0534d0a87e"),
    "ScheherazadeNew-Bold.ttf": ("ofl/scheherazadenew/ScheherazadeNew-Bold.ttf", "91363517a63dbc7448b5814226d33d845b26b32a461cd159ea0b3f23c7effd2d"),
    "ScheherazadeNew.ttf": ("ofl/scheherazadenew/ScheherazadeNew-Regular.ttf", "794bac8dc9e83d1d620bc471ea694f5f31d0965ce8006490a79dfc51a2d283b3"),
    "Tajawal-Bold.ttf": ("ofl/tajawal/Tajawal-Bold.ttf", "0342ab6b74b6bd1b4b5bf7beda45a9734a2b3147a31e8a3a45b6a3417a52d9d7"),
    "Tajawal.ttf": ("ofl/tajawal/Tajawal-Regular.ttf", "6882892da3e03527d5db2bbab3b48bde6ef2e878a43f522d1a4eebda90010a19"),
    "Vazirmatn.ttf": ("ofl/vazirmatn/Vazirmatn[wght].ttf", "696249a2c74b39ffdef55de4df2809c5b639d3ff80d618d8160a095d2fd49dca"),
}


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url) as response:
        return response.read()


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "fonts")
    (out / "LICENSES").mkdir(parents=True, exist_ok=True)
    families = set()
    for name, (path, sha256) in FONTS.items():
        target = out / name
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == sha256:
            continue
        data = _get(RAW + path.replace("[", "%5B").replace("]", "%5D").replace(",", "%2C"))
        if hashlib.sha256(data).hexdigest() != sha256:
            raise SystemExit(f"{path}: not the file this list was made from")
        target.write_bytes(data)
        families.add(path.split("/")[1])
        print("fetched", name)
    for family in sorted(families):
        licence = out / "LICENSES" / f"{family}-OFL.txt"
        if not licence.exists():
            licence.write_bytes(_get(f"{RAW}ofl/{family}/OFL.txt"))
    print(len(FONTS), "fonts in", out)


if __name__ == "__main__":
    main()
