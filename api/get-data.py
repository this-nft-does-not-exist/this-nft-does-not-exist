#!/bin/env python3
import io
import sqlite3
import sys
import numpy
import PIL.Image
import requests
import tqdm
i = 1
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
CREATE_QUERY = "CREATE TABLE nft (image TEXT NOT NULL, data BLOB NOT NULL)"
INSERT_QUERY = "INSERT INTO nft (image, data) VALUES(?, ?)"


def download(cursor, url):
    global i
    res = requests.get(url, headers=HEADERS)
    image = PIL.Image.open(io.BytesIO(res.content)).convert("RGB")
    image.save(f"nft/{i}.png", "PNG")
    i += 1
    # cursor.execute(INSERT_QUERY, (url, numpy.asarray(image).tobytes()))


def main():
    items = []

    items = requests.get(URL, params=PARAMS, headers=HEADERS).json()["sales"]

    with sqlite3.connect(sys.argv[1]) as con:
        cur = con.cursor()
        # cur.execute(CREATE_QUERY)

        for item in tqdm.tqdm(items):
            download(
                cur, f'https://nonfungible.com/_next/image?url=http://images1.nonfungible.com:3333/api/v4/asset/media/image/{item["project"]}/{item["nftTicker"]}/{item["assetId"]}&w=64&q=100')


if __name__ == "__main__":
    main()
