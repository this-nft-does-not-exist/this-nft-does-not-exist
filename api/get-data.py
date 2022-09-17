#!/bin/env python3
import concurrent.futures
import io
import json
import PIL.Image
import requests
import tqdm

CONNECTIONS = 100
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:104.0) Gecko/20100101 Firefox/104.0",
}
URL = "https://nonfungible.com/api/salesForProject"
PARAMS = {
    "filter": '{"blockTimestamp":["2021-04-23T00:00:00.000-04:00","2022-09-17T23:59:59.999-04:00"]}',
    "project": "boredapeclub",
    "limit": 100,
    "orderBy": "blockTimestamp",
    "order": "DESC",
}


def download(url, basename, pbar):
    res = requests.get(url, headers=HEADERS)
    image = PIL.Image.open(io.BytesIO(res.content)).convert("RGB")
    image.save(f"nft/{basename}.png", "PNG")
    pbar.update(1)


def main():
    items = []

    params = dict(PARAMS)
    for i in tqdm.tqdm(range(1000)):
        res = requests.get(URL, params=params, headers=HEADERS).json()["sales"]
        items = items + res
        if res:
            params["after"] = items[-1]["_cursor"]
        with open("items.json", "w") as fd:
            json.dump(items, fd)


    with concurrent.futures.ThreadPoolExecutor(max_workers=CONNECTIONS) as executor:
        with tqdm.tqdm(total=len(items)) as pbar:
            future_to_url = (executor.submit(
                download,
                f'https://nonfungible.com/_next/image?url=http://images1.nonfungible.com:3333/api/v4/asset/media/image/{item["project"]}/{item["nftTicker"]}/{item["assetId"]}&w=64&q=100',
                f'{item["project"]}_{item["nftTicker"]}_{item["assetId"]}',
                pbar
            ) for item in items)
            for future in concurrent.futures.as_completed(future_to_url):
                future.result()


if __name__ == "__main__":
    main()
