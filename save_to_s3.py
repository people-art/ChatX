# Save the *.pdf.pkl file to S3 bucket.

import os
import boto3
from tqdm import tqdm

os.environ["AWS_ACCESS_KEY_ID"] = "AKIA5TRYXY54DLEBWG4X"
os.environ["AWS_SECRET_ACCESS_KEY"] = "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"

s3 = boto3.client("s3")
bucket_name = "carbonchat"


print("Uploading *.pdf.pkl files...")
pkl_files = [f for f in os.listdir("pkl_files") if f.endswith(".pkl")]
for filename in tqdm(pkl_files, desc="Uploading pkl files"):
    s3.upload_file(os.path.join("temp", filename), bucket_name, os.path.join("pkl_files", filename))