import requests

IAM_TOKEN = 't1.9euelZqXnJCSjJuQxo3HlJaNisaLyu3rnpWayZeKkJ6bxo-ej5yOjc6Syo_l8_cxXgps-e92Zj5x_d3z93EMCGz573ZmPnH9.UWi20GqmHoso67A61i_K6VGWQqCgbdDyKvoGoAzUpcCv3sekEG-UKcST0yggmcdJua7cQfWcm6-6obEH4Z_wBg'
folder_id = 'b1gr1ji0opfu3edba2hi'
target_language = 'ru'


def translate(texts):

    body = {
        "targetLanguageCode": target_language,
        "texts": texts,
        "folderId": folder_id,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer {0}".format(IAM_TOKEN)
    }

    response = requests.post('https://translate.api.cloud.yandex.net/translate/v2/translate',
                             json=body,
                             headers=headers
                             )
    return response.json()['translations']

if __name__ == "__main__":
    print(translate(['안녕하세요. 어떻게 지내세요?','지내세요?']))