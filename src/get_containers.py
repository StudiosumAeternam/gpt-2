from azure.storage.blob import BlobServiceClient

storage_account_url = "https://openaipublic.blob.core.windows.net/"

# Используйте анонимный доступ
blob_service_client = BlobServiceClient(account_url=storage_account_url, credential=None)

# Получить список контейнеров
containers_list = blob_service_client.list_containers()

# Вывести список контейнеров
for container in containers_list:
    print(container.name)
