from pathlib import Path
import zipfile
from azure.storage.queue import (
    QueueClient,
    BinaryBase64EncodePolicy,
    BinaryBase64DecodePolicy
)

import os
import requests
import tempfile
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__
import json


class Azure_Helper(object):
    def __init__(self):
        print("Azure Helper....")
        #
        # variable named AZURE_STORAGE_CONNECTION_STRING
        self.containers_queues_connect_str = os.getenv(
            "AZURE_STORAGE_CONNECTION_STRING")
        self.scans_storage_connect_str = os.getenv(
            "AZURE_SCANS_STORAGE_CONNECTION_STRING")
        self.queue_name = "segmentation"

    def get_next_message(self):
        # Instantiate a QueueClient object which will
        # be used to create and/or manipulate the queue
        # self.queue_client = QueueClient.from_connection_string(self.connect_str, self.queue_name)
        # Setup Base64 encoding and decoding functions
        queue_client = QueueClient.from_connection_string(
            conn_str=self.containers_queues_connect_str, queue_name=self.queue_name,
            message_encode_policy=BinaryBase64EncodePolicy(),
            message_decode_policy=BinaryBase64DecodePolicy()
        )
        messages = queue_client.receive_messages()
        for message in messages:
            queue_client.delete_message(message.id, message.pop_receipt)
            return message
        return None

    def get_container(self, number):
        container = str(number)
        if container.startswith('20') and len(container) >= 7:
            return container
        else:
            return "20" + container

    def get_order_number(self, number):
        container = str(number)
        if container.startswith('20') and len(container) >= 7:
            return container[2:]
        else:
            return container

    def get_azure_blob(self, order):
        # Instantiate a BlobServiceClient object which will be used to create a container client
        blob_service_client = BlobServiceClient.from_connection_string(
            conn_str=self.containers_queues_connect_str)
        # Ajust the container name from the order
        container_name = self.get_container(order)

        container = blob_service_client.get_container_client(container_name)

        # List the blobs in the container
        blob_list = container.list_blobs()
        for blob in blob_list:
            print("\t Blob name: " + blob.name)
            if blob.name.upper().endswith(".3sz".upper()) and "exported_3shape" in blob.name:
                temp_file = tempfile.NamedTemporaryFile()
                temp_file = temp_file.name + ".zip"
                with open(temp_file, "wb") as file_obj:
                    file_content = container.get_blob_client(
                        blob).download_blob().readall()
                    file_obj.write(file_content)
                return temp_file
        return None

    def zip_and_upload(self, path, order):
        if not os.path.exists(path):
            return
        # zip all files in path
        root_path = Path(path).parent
        zip_file = f"{root_path}/arcseg.sgt"
        seg_path = os.path.join(path, "segmentation")
        with zipfile.ZipFile(zip_file, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip:
            for root, dirs, files in os.walk(seg_path):
                for file in files:
                    zip.write(os.path.join(root, file), arcname=file)
        # upload zip file to azure
        # get the order number
        #order_container = Path(path).name
        # Instantiate a BlobServiceClient object which will be used to create a container client
        blob_service_client = BlobServiceClient.from_connection_string(
            conn_str=self.containers_queues_connect_str)
        # Ajust the container name from the order
        container_name = self.get_container(order)
        container = blob_service_client.get_container_client(container_name)
        blob_name = f"segmentation/{Path(zip_file).name}"
        # delete blob if exits
        try:
            blob = container.get_blob_client(blob_name)
            blob.delete_blob()
        except:
            pass
        print(f"Uploading {zip_file} to {container_name}/{blob_name}")
        with open(zip_file, "rb") as data:
            container.upload_blob(blob_name, data, timeout="300")
        return blob_name

    # work in progress.....
    def get_order_original_scans(self, ordernum):
        url = f"https://theportal.arcadlab.com/api/public/getorderscans?orderpublicId={ordernum}"
        print(f"Getting order {ordernum} scans")
        request = requests.get(url)
        if request.status_code == 200:
            result = {}
            scans = json.loads(request.text)
            tmp = tempfile.gettempdir()
            tmpdirname = os.path.join(tmp, "AViewer", str(
                self.get_container(ordernum)), "scans")
            result['working_dir'] = tmpdirname
            os.makedirs(tmpdirname, exist_ok=True)
            scan_lower = requests.get(scans['lower'])
            lower_path = os.path.join(tmpdirname, "scan_lower.stl")
            if os.path.exists(lower_path):
                os.remove(lower_path)  # this deletes the file
            with open(lower_path, 'wb')as file:
                file.write(scan_lower.content)
            result["lower"] = lower_path

            scan_upper = requests.get(scans['upper'])
            upper_path = os.path.join(tmpdirname, "scan_upper.stl")
            # if os.path.exists(upper_path):
            #    os.remove(upper_path)  # this deletes the file
            with open(upper_path, 'wb')as file:
                file.write(scan_upper.content)
            result["upper"] = upper_path

            return result
        return None
