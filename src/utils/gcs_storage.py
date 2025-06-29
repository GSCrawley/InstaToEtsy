import os
import logging
from typing import Optional, Dict, Any, List
from google.cloud import storage
from google.oauth2 import service_account
from .. import config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GCSStorage:
    def __init__(self):
        """Initialize GCS client using credentials from config."""
        try:
            # Check if credentials path is set
            if not config.GOOGLE_APPLICATION_CREDENTIALS:
                logger.warning("GCS credentials path not set. GCS operations will not work.")
                self.client = None
                self.bucket = None
                return
                
            # Check if credentials file exists
            if not os.path.exists(config.GOOGLE_APPLICATION_CREDENTIALS):
                logger.warning(f"GCS credentials file not found at {config.GOOGLE_APPLICATION_CREDENTIALS}. GCS operations will not work.")
                self.client = None
                self.bucket = None
                return
                
            # Initialize GCS client
            credentials = service_account.Credentials.from_service_account_file(
                config.GOOGLE_APPLICATION_CREDENTIALS
            )
            self.client = storage.Client(credentials=credentials, project=config.GCS_PROJECT_ID)
            
            # Get the bucket
            if not config.GCS_BUCKET_NAME:
                logger.warning("GCS bucket name not set. GCS operations will not work.")
                self.bucket = None
                return
                
            self.bucket = self.client.bucket(config.GCS_BUCKET_NAME)
            logger.info(f"Successfully initialized GCS client for bucket {config.GCS_BUCKET_NAME}")
        except Exception as e:
            logger.error(f"Error initializing GCS client: {e}")
            self.client = None
            self.bucket = None
            
    def is_available(self) -> bool:
        """Check if GCS client is available and properly configured."""
        return self.client is not None and self.bucket is not None
        
    def upload_file(self, source_file_path: str, destination_blob_name: str) -> bool:
        """
        Upload a file to GCS bucket.
        
        Args:
            source_file_path: Path to the local file to upload.
            destination_blob_name: Name to give the file in GCS.
            
        Returns:
            True if upload was successful, False otherwise.
        """
        if not self.is_available():
            logger.error("GCS client not available. Cannot upload file.")
            return False
            
        try:
            blob = self.bucket.blob(destination_blob_name)
            blob.upload_from_filename(source_file_path)
            logger.info(f"File {source_file_path} uploaded to {destination_blob_name}.")
            return True
        except Exception as e:
            logger.error(f"Error uploading file to GCS: {e}")
            return False
            
    def upload_from_string(self, data: str, destination_blob_name: str, content_type: str = 'text/plain') -> bool:
        """
        Upload data from a string to GCS bucket.
        
        Args:
            data: String data to upload.
            destination_blob_name: Name to give the file in GCS.
            content_type: MIME type of the data.
            
        Returns:
            True if upload was successful, False otherwise.
        """
        if not self.is_available():
            logger.error("GCS client not available. Cannot upload data.")
            return False
            
        try:
            blob = self.bucket.blob(destination_blob_name)
            blob.upload_from_string(data, content_type=content_type)
            logger.info(f"Data uploaded to {destination_blob_name}.")
            return True
        except Exception as e:
            logger.error(f"Error uploading data to GCS: {e}")
            return False
            
    def download_file(self, source_blob_name: str, destination_file_path: str) -> bool:
        """
        Download a file from GCS bucket.
        
        Args:
            source_blob_name: Name of the file in GCS.
            destination_file_path: Path to save the file locally.
            
        Returns:
            True if download was successful, False otherwise.
        """
        if not self.is_available():
            logger.error("GCS client not available. Cannot download file.")
            return False
            
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)
            
            blob = self.bucket.blob(source_blob_name)
            blob.download_to_filename(destination_file_path)
            logger.info(f"File {source_blob_name} downloaded to {destination_file_path}.")
            return True
        except Exception as e:
            logger.error(f"Error downloading file from GCS: {e}")
            return False
            
    def list_files(self, prefix: str = '') -> List[str]:
        """
        List files in the GCS bucket with the given prefix.
        
        Args:
            prefix: Prefix to filter files by.
            
        Returns:
            A list of file names in the bucket matching the prefix.
        """
        if not self.is_available():
            logger.error("GCS client not available. Cannot list files.")
            return []
            
        try:
            blobs = self.client.list_blobs(self.bucket, prefix=prefix)
            return [blob.name for blob in blobs]
        except Exception as e:
            logger.error(f"Error listing files in GCS: {e}")
            return []
            
    def file_exists(self, blob_name: str) -> bool:
        """
        Check if a file exists in the GCS bucket.
        
        Args:
            blob_name: Name of the file in GCS.
            
        Returns:
            True if the file exists, False otherwise.
        """
        if not self.is_available():
            logger.error("GCS client not available. Cannot check if file exists.")
            return False
            
        try:
            blob = self.bucket.blob(blob_name)
            return blob.exists()
        except Exception as e:
            logger.error(f"Error checking if file exists in GCS: {e}")
            return False
            
    def delete_file(self, blob_name: str) -> bool:
        """
        Delete a file from the GCS bucket.
        
        Args:
            blob_name: Name of the file in GCS.
            
        Returns:
            True if deletion was successful, False otherwise.
        """
        if not self.is_available():
            logger.error("GCS client not available. Cannot delete file.")
            return False
            
        try:
            blob = self.bucket.blob(blob_name)
            blob.delete()
            logger.info(f"File {blob_name} deleted from GCS.")
            return True
        except Exception as e:
            logger.error(f"Error deleting file from GCS: {e}")
            return False
