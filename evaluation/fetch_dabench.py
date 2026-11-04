"""Fetch the pinned public InfiAgent-DABench snapshot, never execute it.

Run once before evaluation, or use --offline to verify a previously fetched
snapshot without any network requests. Only public immutable HTTPS resources
are accessed. Downloaded Python remains inert source text outside src/.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / ".cache/evaluation/dabench-source"
DATASET = "infiagent/DABench"
DATASET_REVISION = "b455d578e30fee513abd79936cbdf7a6de026cb5"
SCORER_REPOSITORY = "InfiAgent/InfiAgent"
SCORER_REVISION = "3d6c4a70198e0a41fadf539f5b43c88b8c1a2d9c"
SCORER_SHA256 = "8c4cbeed80c6c95aca65f72cea3213581629e81894a0fa5a496f22726b6e4501"
ALLOWED_HOSTS = frozenset({"huggingface.co", "raw.githubusercontent.com"})

# Audited bytes at the immutable dataset revision, not freshly claimed hashes.
# The 70 entries contain questions, labels and all 68 original CSV files.
DATA_FILES = (
    ("da-dev-questions.jsonl", "430306c5813bf4f2b0e38f1b6ccee9693c36d5920b3322bd7c8ae46f475ce443", 217261),
    ("da-dev-labels.jsonl", "83b8fb8133c1794fcb6b42435e9cce124d22f68b71e2eedb37c973e40b5a18c2", 23680),
    ("da-dev-tables/0020200722.csv", "d9b292cecc04c3993caf084927a1318feb990e641accc966d367eedc6d9c349a", 32429),
    ("da-dev-tables/2014_q4.csv", "f1ec88490c4cf2fde96a2783f2962743a126e66e59b16d83378192941b5dbd82", 5615),
    ("da-dev-tables/2015-09-21.csv", "14baf261342a0cb78a00e9b81273a111259746fdc7e7b17ded48139a0c250447", 5507399),
    ("da-dev-tables/2015.csv", "d79be64d11b4e36159242055e028e29ee6b4c3da6838e62d7a925bdde26eab5d", 16557),
    ("da-dev-tables/20170413_000000_group_statistics.csv", "17da5edda18a5a5c2168f5f06a0a90604efef810cac0d17492723ddad2685753", 5692),
    ("da-dev-tables/2019-08_edges.csv", "3511be9a3e685f9416de3257f6f95149e80fffce042c14bce5ffc161a43140cc", 1084430),
    ("da-dev-tables/3901.csv", "86e7e9f8c381c36441f72800f20c011497351541e1e9a9d3f532844922c447ae", 220784),
    ("da-dev-tables/Baltimore_City_Employee_Salaries_FY2013.csv", "f18a76062c34423522ec1d167ea3f47a633afebb7e28f8f4944003b3cf2c6bbb", 1834913),
    ("da-dev-tables/Credit.csv", "eb75b3d418d190a4b71f01a8333c49819928591b5a0e1f595546f1987358f875", 21967),
    ("da-dev-tables/Current_Logan.csv", "045a58d23610362327912f887c6bf24e0ddde01a46bcf1f2ea8acda47e05b350", 2405),
    ("da-dev-tables/DES=+2006261.csv", "753bcb0a243256986b0b663666d0bbf5dd273f0de753ad0705be1c9cbacf8f83", 40500),
    ("da-dev-tables/GODREJIND.csv", "55770473647050ba8f11d46108aac81f38df600e8261b105321fffd5c5a6ecec", 116303),
    ("da-dev-tables/Gillam_fin.csv", "df805d319b595b40cd970a6ab1b5a7d1298ca9fb5a8d1ec6e43f373a7c051b74", 2662229),
    ("da-dev-tables/Todo_Tango_letras_final.csv", "016ac07e8311717cc0541b887257ea39c31d6a75874cbf2ab580c14e809e4f76", 6279530),
    ("da-dev-tables/YAHOO-BTC_USD_D.csv", "ddbd797b7db4999bbb7e836ad928aba56bea7a5dbeb516afb7d63fd42662a328", 173887),
    ("da-dev-tables/Yelp_Reviews.csv", "a2a67c43b07dcb8e00445fc5a8f0e0111f680aacfac6f28f695ec92de8e3d14d", 1335192),
    ("da-dev-tables/Zip_MedianSoldPricePerSqft_AllHomes.csv", "f4dac692bdc69c427a63fe3e77eda2c7ca07c7eb1bdb52a3061b0b91c66f4b7b", 394007),
    ("da-dev-tables/abalone.csv", "6fda3c4b0c0cff9bd5f7a46434509179069ed620c8fc7090d872a08d2c1a729a", 191962),
    ("da-dev-tables/api_output_2018-09-11_11-08-28.csv", "6c04dc92a55034d1a44c65aaddf63a9b3115a9467f679b5e66500c9c766595a5", 2650914),
    ("da-dev-tables/arrest_expungibility.csv", "c985ca12445770b8559667f3a4c8d18166571b114f2d1c4dc2ac1b3a56c94b39", 670664),
    ("da-dev-tables/auto-mpg.csv", "555a2b668da52d7431215df842fb073d1ad220c56d4ac6ec359c0ae9b1beea6b", 14179),
    ("da-dev-tables/baro_2015.csv", "34919769cf48ec84fc6358cc1c2f27b56a4e624af293f3ffcd7531f00bcc1d8a", 382752),
    ("da-dev-tables/baseball_data.csv", "ffcf9e1931e1fa882782df55c638d8f698a0487c7048126bfcefe088f22e4cf7", 21647),
    ("da-dev-tables/beauty and the labor market.csv", "dba6fb21faf1fd88e99f8e55770de819d7cb6db213780d13a42e2f8b3289dbca", 64937),
    ("da-dev-tables/bike_rental_hour.csv", "b03a2d02e8c10f435c43c7f0b358b7e34a003afea53dbc37f0183f2763295133", 1139356),
    ("da-dev-tables/bitconnect_price.csv", "c970dd13f2e735fc9a654f69ef6c9669ff2d0466f01eccee000d1dc6e24f5bc2", 15668),
    ("da-dev-tables/census.csv", "6dbe6e310376b728cf0c27eb15925e431eb91ad4df35343661dfffec9c606aee", 3974464),
    ("da-dev-tables/city_departments_in_current_budget.csv", "8fcaca49cb80ea8345bcee68d545c2046b016de71f5989cd7459a451b75ece51", 2677),
    ("da-dev-tables/cms_hospital_readmissions.csv", "f819152e7f40f2d48640ece35b251fdacd746b4bde32eb019adfabb5dbbf3f28", 1773532),
    ("da-dev-tables/cost_data_with_errors.csv", "83efce074700d226453f08c9b8faddc4703c70ae0e1888e8e8422994416d1ce0", 82522),
    ("da-dev-tables/country_vaccinations.csv", "eb38b5933c1dafbd884a9042c23afc5181bedf6c434d941115444284ea215fcc", 680231),
    ("da-dev-tables/credit-data-post-import.csv", "4560780d211a2cda8cbd624a6e375c1682a1f0b87237788cd78b2b3710f5f099", 8779642),
    ("da-dev-tables/cs-test.csv", "5a0b087d1bf62933417ee483425a193f828bd94b229fb6715d5eac016713f597", 4881825),
    ("da-dev-tables/diamonds.csv", "9574730b03aba241d899c4a97511c5061b19358fab89510774fb6c24168345c4", 2772143),
    ("da-dev-tables/e5_aapl.csv", "d81f47727583df630307cc7d648d0f30ae5fea79fe9ff8ad799db344c36f7ae2", 34849),
    ("da-dev-tables/election2016.csv", "b2d37e2153fa661b710170db1c9b2d62cef2baf2910535bd5600c32892f48a45", 291557),
    ("da-dev-tables/estimated_numbers.csv", "bc485997a837c738907ac5b03aef0f4d0a938edb415861ab37af22bdd243329c", 66590),
    ("da-dev-tables/fb_articles_20180822_20180829_df.csv", "506726b8aa93fd36bdb626a4c5fa702295a2207e6b2536e82f738d9cc2522050", 927008),
    ("da-dev-tables/ferret-Pitt-2-preinf-lib2-100_sitediffsel.csv", "c9e411ff75df8d966f6e5de9270a4a7276289c478a6b7aa216ce667eac66f540", 57084),
    ("da-dev-tables/gapminder_cleaned.csv", "9e6f0a924da3b91a0c7ba24260f773f8648deb2475ca4bbd5d16f1366a0b45a7", 82019),
    ("da-dev-tables/gapminder_gdp_asia.csv", "d59ff928054c78dfb6613046d306fbc044ef754c6ca3b2aff71b661a240f869e", 5103),
    ("da-dev-tables/hotel_data.csv", "581144a1a813e187abc75c3db8f71c25cbec27afe7956e9be9de369acde1c0f7", 83961),
    ("da-dev-tables/housedata.csv", "1632d72fa14167a800f0fc1b9ef66a0cad9f2d31653329909e4a8979377a5f06", 2049273),
    ("da-dev-tables/imp.score.ldlr.metabolome.csv", "8aa6112a864d4491ec14b7821f0ce889a6f751793ec30c7b9d86c5443e25c553", 55375),
    ("da-dev-tables/insurance.csv", "6e3894cf7d9210613f62c866684055decea694345cd4f244ecd7167041578c68", 54289),
    ("da-dev-tables/microsoft.csv", "6d0f08343d42a5ad45d1eb2275a99416aa8a0b26be03fe5e74538cff31352292", 11515),
    ("da-dev-tables/my_test_01.csv", "3134c6c7260510da5aaa33441743d7aaa52efd88fc4d443e7c7a98f80d734c25", 23732),
    ("da-dev-tables/oecd_education_spending.csv", "68a9ce2a7b5c249f01c716d92eb5f7429be1c6185f0160a6b73d6df89e425419", 9285),
    ("da-dev-tables/percent-bachelors-degrees-women-usa.csv", "4f3a2a6a6b15fcddddde55bb48a9a3d78cd9557e0538e83d8c972c0b953da5b0", 5681),
    ("da-dev-tables/random_stock_data.csv", "425acc100e1a260e1206d7f12aa03e5186a3895a4c4fa2da5b3116f1350d2f76", 30299),
    ("da-dev-tables/ravenna_250715.csv", "323831969f08919961b36bb8f6b06f06493b25b01a5cf4fb649f7b39224df7de", 2306),
    ("da-dev-tables/test_Y3wMUE5_7gLdaTN.csv", "2248e254edd4260a5cbae11856115d3d0336396239188e970c18c6e12dda4cd8", 21589),
    ("da-dev-tables/test_ave.csv", "411cf03455d6026823fbd3ab65e2839075a22f9a5c088b85aef0d272d79cca00", 55169),
    ("da-dev-tables/test_x.csv", "be19636b5bb50967a295060f62a26d7911a746527d6d946b0994165bbfade659", 6741),
    ("da-dev-tables/titanic.csv", "4a437fde05fe5264e1701a7387ac6fb75393772ba38bb2c9c566405af5af4bd7", 60302),
    ("da-dev-tables/titanic_test.csv", "5c78320a80dc159b35c408363be46a5d93f078183d539db75ee42a796d6a4e95", 28210),
    ("da-dev-tables/titanic_train.csv", "4a437fde05fe5264e1701a7387ac6fb75393772ba38bb2c9c566405af5af4bd7", 60302),
    ("da-dev-tables/tr_eikon_eod_data.csv", "eb59a911fcc95b0b7fd5df55ce9349850dd20e6ae9c8efc9e5db308151c9a5f4", 209046),
    ("da-dev-tables/train_new_ocr_29.csv", "a652fa7275522374d7ea8fbfb82468e1ef94f5fd7ff3a7d38421a3435e2ba2e5", 3018280),
    ("da-dev-tables/traj-Osak.csv", "a7c439f3f74f6f096579a66fe17d191f600556bacdf4d15060589a4f36912691", 67251),
    ("da-dev-tables/tree.csv", "2d0abdcd475f0394b8f2f106faf14e1ae86887c20c3d23bb0541e1a20ef73129", 699211),
    ("da-dev-tables/ts-sc4-wi100000-sl25000-Qrob_Chr05.tree_table.csv", "3e0540386cf89b4f5e1d7f2760f9aa3885ea21119d3b12e9e5c07056caae7367", 2295032),
    ("da-dev-tables/unemployement_industry.csv", "c6fce742e7177c3ea07f5080dac8a54fcf8069c9bfab11012a4d9e5e1ac94dd9", 4185),
    ("da-dev-tables/veracruz 2016.csv", "2890f10a92b706c260de39281a814442249716ea757029ae5425f7e12ff1e11b", 819397),
    ("da-dev-tables/vgsales.csv", "e2076095ffcae2a92dbc6de6ecbd54455ee034bb38da550b2dc385e9265d4ebe", 1355781),
    ("da-dev-tables/weather_data_1864.csv", "6bd35d0c907e8c70ba15cced1d773e0e31b7bfff9f4251719e7740111958aef9", 285512),
    ("da-dev-tables/weather_train.csv", "c9bd23cabbba04970b574259b62025c9ac08ef103ae760f66e353b73116620c0", 840734),
    ("da-dev-tables/well_2_complete.csv", "14d21b201758300d20e030feb564f67a21d85205d29b866587db9017c83880b8", 695333),
)
SCORER_FILES = (
    ("eval_closed_form.py", "examples/DA-Agent/eval_closed_form.py",
     SCORER_SHA256, 8634),
    ("reformat.py", "examples/DA-Agent/reformat.py",
     "0c030134b6b24e98f916f1d6178f5d1d2ae0e295d170634175c8829c37f7121b", 3530),
    ("README.md", "examples/DA-Agent/README.md",
     "bc465febedfe4d8ea0ae7dcb2c41f34889ef4bdba52a97d734f2b0b55feeadd6", 7614),
)


def _validate_url(url, *, redirect=False):
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS
            or parsed.username or parsed.password or parsed.port not in (None, 443)
            or (parsed.query and not redirect) or parsed.fragment):
        raise ValueError("Only fixed public HTTPS source hosts are allowed")
    return url


class _OfficialRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        # Hugging Face resolve-cache redirects can include public ETag query
        # parameters. They remain HTTPS on the fixed official host allowlist.
        _validate_url(new_url, redirect=True)
        return super().redirect_request(request, fp, code, message, headers, new_url)


def _public_bytes(url, expected_size):
    """Read no more than the pinned size; redirects cannot escape source hosts."""
    request = Request(_validate_url(url), headers={"User-Agent": "AnaliyMaster-benchmark-fetch/1"})
    opener = build_opener(_OfficialRedirects())
    with opener.open(request, timeout=60) as response:
        raw = response.read(expected_size + 1)
    if len(raw) != expected_size:
        raise ValueError("Official resource size differs from the pinned snapshot")
    return raw


def _is_link(path):
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _checked_root(output):
    root = Path(output).absolute()
    # Inspect before resolve(): resolve alone would erase evidence of a link.
    for candidate in [*reversed(root.parents), root]:
        if candidate.exists() or candidate.is_symlink():
            if _is_link(candidate):
                raise ValueError("Download directories cannot contain links or reparse points")
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise ValueError("Output must be a directory")
    return root.resolve()


def _target(root, relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError("Unsafe relative download path")
    path = PurePosixPath(relative)
    if (not path.parts or path.is_absolute()
            or relative != path.as_posix() or any(p in {"", ".", ".."} for p in path.parts)
            or "\\" in relative or ":" in relative or "\x00" in relative):
        raise ValueError("Unsafe relative download path")
    target = root.joinpath(*path.parts)
    if not target.is_relative_to(root):
        raise ValueError("Download path escapes the output directory")
    for candidate in [*reversed(target.parents), target]:
        if not candidate.is_relative_to(root):
            continue
        if candidate.exists() or candidate.is_symlink():
            if _is_link(candidate):
                raise ValueError("Downloaded files cannot contain links or reparse points")
    if not target.resolve().is_relative_to(root):
        raise ValueError("Resolved download path escapes the output directory")
    return target


def _verify_file(path, sha256, expected_size):
    if not path.is_file() or path.stat().st_size != expected_size:
        raise ValueError("Cached resource is missing or has an unexpected size: " + path.name)
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    if hasher.hexdigest() != sha256:
        raise ValueError("Cached resource SHA-256 mismatch: " + path.name)


def _ensure_file(root, relative, url, sha256, expected_size, offline):
    target = _target(root, relative)
    if target.exists():
        _verify_file(target, sha256, expected_size)
        return "reused"
    if offline:
        raise ValueError("Offline resource missing: " + relative)
    raw = _public_bytes(url, expected_size)
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("Official resource SHA-256 differs from pinned snapshot: " + relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    target = _target(root, relative)
    created = False
    try:
        with target.open("xb") as stream:
            created = True
            stream.write(raw)
    except FileExistsError:
        _verify_file(target, sha256, expected_size)
        return "reused"
    except BaseException:
        if created:
            target.unlink(missing_ok=True)
        raise
    return "downloaded"


def _manifest(root, name, expected):
    path = _target(root, name)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != expected:
            raise ValueError("Source manifest differs from the pinned snapshot: " + name)
        return  # Keep original bytes, preserving downstream frozen fingerprints.
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(expected, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def fetch_dataset(output=DEFAULT_OUTPUT, *, offline=False, workers=4):
    """Verify/reuse all pinned resources; download only missing valid resources."""
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("workers must be an integer from 1 through 4")
    root = _checked_root(output)
    data_manifest = {
        "dataset": DATASET, "revision": DATASET_REVISION,
        "files": [{"path": p, "sha256": sha, "size": size} for p, sha, size in DATA_FILES],
        "download_errors": [],
    }
    scorer_manifest = {
        "repository": SCORER_REPOSITORY, "revision": SCORER_REVISION,
        "files": [{"path": p, "sha256": sha, "size": size} for p, _, sha, size in SCORER_FILES],
    }
    # Reject a changed manifest before making network requests or writing data.
    for name, expected in (("source.json", data_manifest), ("scorer-source.json", scorer_manifest)):
        path = _target(root, name)
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != expected:
            raise ValueError("Source manifest differs from the pinned snapshot: " + name)
    resources = [
        (p, f"https://huggingface.co/datasets/{DATASET}/resolve/{DATASET_REVISION}/{quote(p, safe='/')}", sha, size)
        for p, sha, size in DATA_FILES
    ]
    resources += [
        (f"upstream-{p}", f"https://raw.githubusercontent.com/{SCORER_REPOSITORY}/{SCORER_REVISION}/{quote(remote, safe='/')}", sha, size)
        for p, remote, sha, size in SCORER_FILES
    ]
    def ensure(spec):
        return _ensure_file(root, *spec, offline)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        statuses = list(pool.map(ensure, resources))
    _manifest(root, "source.json", data_manifest)
    _manifest(root, "scorer-source.json", scorer_manifest)
    return {
        "dataset": DATASET, "revision": DATASET_REVISION,
        "scorer_revision": SCORER_REVISION, "scorer_sha256": SCORER_SHA256,
        "data_files": len(DATA_FILES),
        "csv_files": sum(p.startswith("da-dev-tables/") for p, _, _ in DATA_FILES),
        "downloaded": statuses.count("downloaded"), "reused": statuses.count("reused"),
        "offline": offline, "output": str(root),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--offline", action="store_true", help="Verify cached bytes without network access")
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    args = parser.parse_args()
    print(json.dumps(fetch_dataset(args.output, offline=args.offline, workers=args.workers), ensure_ascii=False))


if __name__ == "__main__":
    main()
