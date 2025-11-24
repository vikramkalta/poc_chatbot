import requests

BASE_URL = "https://staging.borga.money/borga/user/api/v1/devices"
AUTH_TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJtUFpKMEMyQUphZVl3TU5PQm9sSU9pRVVocy1INlhRUkh6SDlUcXVjekxZIn0.eyJleHAiOjE3NjYyNTQ0ODAsImlhdCI6MTc2MzY2MjQ4MCwianRpIjoiNTVmYTM3M2ItYTdkNi00YWMyLWEzMzEtYTAwNGQ0MzVmMzFiIiwiaXNzIjoiaHR0cHM6Ly9zdGFnaW5nLmJvcmdhLm1vbmV5L3JlYWxtcy9ib3JnYS1kZXYiLCJhdWQiOiJhY2NvdW50Iiwic3ViIjoiMTEwYjExOGMtYWM0MS00MDQ1LTkzNGMtMTIzMjMwYTE0ZTVjIiwidHlwIjoiQmVhcmVyIiwiYXpwIjoidXNlcl9zZXJ2aWNlIiwic2Vzc2lvbl9zdGF0ZSI6IjIwNDcxMzg4LTAyNWMtNDIzYi1iZTRkLTM5NzNkZTQzZjA0OCIsImFjciI6IjEiLCJyZWFsbV9hY2Nlc3MiOnsicm9sZXMiOlsiYm9yZ2FfY3JlYXRlX21lcmNoYW50IiwiYm9yZ2Ffc2V0X2ZvcmV4X3JhdGUiLCJib3JnYV91cGRhdGVfbWVyY2hhbnQiLCJib3JnYV92aWV3X3RyYW5zYWN0aW9ucyIsImJvcmdhX3VwZGF0ZV9kZWxpdmVyeV9hY2NvdW50IiwiYm9yZ2Ffdmlld19hdWRpdF90cmFpbCIsImRlZmF1bHQtcm9sZXMtYm9yZ2EtZGV2IiwiYm9yZ2FfbWFuYWdlX2RlbGl2ZXJpZXMiLCJvZmZsaW5lX2FjY2VzcyIsImJvcmdhX2Rpc3B1dGVfcmVzb2x1dGlvbiIsImJvcmdhX2NyZWF0ZV9kZWxpdmVyeV9hY2NvdW50IiwiYm9yZ2FfY3JlYXRlX2VtcGxveWVlIiwiYm9yZ2FfdXBkYXRlX2FkbWluIiwidW1hX2F1dGhvcml6YXRpb24iLCJib3JnYV9yZXBvcnRpbmciLCJib3JnYV9tYW5hZ2Vfb3JkZXJzIiwiYm9yZ2FfdXBkYXRlX2VtcGxveWVlIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwic2NvcGUiOiJlbWFpbCBwcm9maWxlIiwic2lkIjoiMjA0NzEzODgtMDI1Yy00MjNiLWJlNGQtMzk3M2RlNDNmMDQ4IiwiZW1haWxfdmVyaWZpZWQiOmZhbHNlLCJuYW1lIjoiRGl2aW5lIEV0ZWJhIiwicHJlZmVycmVkX3VzZXJuYW1lIjoiMjMzNTQzMzA4NDYyQG4vYS5ib3JnYSIsImdpdmVuX25hbWUiOiJEaXZpbmUiLCJmYW1pbHlfbmFtZSI6IkV0ZWJhIiwiZW1haWwiOiJldGViYWRpdmluZTM1OEBnbWFpbC5jb20ifQ.ORfBV6Qt-z5aKT1aRj41TLejliqR6AUxa2mk4xlDBjDVA6f_0mN2aZ_tkP10ReWONJSAWjfYuoQBWLv4iaw0hMkJ-7UecPJOYhVIe9FO2619aPvs6Te0BTdnMdfSW9pb58jReugFWky0mgf3iCaaE23m9EYYepDxXgF0c5nIoxCaThEeINugYIouwsWpIsPSjoGdf2Zder8usmqIGRvssMfqbKdDP0ws_7WXd2DISoeX0tEet3H4wP7ojVE7c2hc_3ESVsY84dAwtnYSwrz2lFds8b0MWj6_XlUm_-e2VjJY5uhtm9RRZhYrCZAozWo0Yl6fCrLgYPZITWLrT89GBg"   # <-- replace with real token

# List of device IDs to delete
device_ids = [
    "012b56cd-d687-4fe0-ac7f-12a3395928e3",
    "567232a7-7a3c-4f6b-aca1-c2bad23103a1",
    "63557bd0-f3a5-409e-87ac-80fb97fe07c7",
    "6bb71ed2-2d2b-488a-9779-2dcebbc2e834",
    "7059cdf6-a493-4c28-b4ce-5b5e89e4da89",
    "71f67dc2-9bbe-49f9-8dec-9e18a9e76e07",
    "75021a2c-1f3b-40f4-97eb-27680a65e7d1",
    "7c873226-aaf0-486b-8fa7-0d502ade9cf2",
    "8a018798-277a-4fdc-89a9-9770deb48d21",
    "9391ad05-278c-46c6-a1dd-0d5f6b0cf6f2",
    "954ec633-f36e-4951-8679-026c5651b8b6",
    "a52f174a-8868-468c-9308-de56bcd90e78",
    "adf6072b-bbbe-4c35-9028-9703c2aa7ab2",
    "dc613608-1b4c-4aaa-90fe-09e2668df64f",
    "de5a712c-7b39-4b7e-91e8-85f42bd29aa5",
]

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AUTH_TOKEN}"
}

for device_id in device_ids:
    url = f"{BASE_URL}/{device_id}"
    response = requests.delete(url, headers=headers)

    print(f"Deleting {device_id} -> Status: {response.status_code}")
    try:
        print("Response:", response.json())
    except:
        print("No JSON response")