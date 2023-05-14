from azure.storage.blob import BlobServiceClient, ContainerClient

# Замените следующие значения на соответствующие значения вашего случая
account_url = "https://openaipublic.blob.core.windows.net/"
container_name = "gpt-2"

# Создайте клиент BlobService
blob_service_client = BlobServiceClient(account_url=account_url)

# Создайте клиент контейнера
container_client = blob_service_client.get_container_client(container_name)

# Получите список всех файлов (Blobs) в контейнере
blob_list = container_client.list_blobs()

# Отсортируйте список файлов по времени последнего изменения в порядке убывания (от самого нового к самому старому)
sorted_blob_list = sorted(blob_list, key=lambda x: x.last_modified, reverse=True)

# Выведите имена всех файлов (Blobs) и время их последнего изменения
for blob in sorted_blob_list:
    print(f"File name: {blob.name}, Last modified: {blob.last_modified}")