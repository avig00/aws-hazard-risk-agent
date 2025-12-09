import boto3
import json

def predict_from_endpoint(payload: dict, endpoint_name: str):
    """
    Call SageMaker endpoint and return predictions.
    """
    sm = boto3.client("runtime.sagemaker")
    response = sm.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload)
    )
    prediction = json.loads(response["Body"].read())
    return prediction

if __name__ == "__main__":
    print("Inference service placeholder")
