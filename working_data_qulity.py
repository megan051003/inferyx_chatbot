import json
import requests
from typing import Optional, Any, Dict, List
from box import ConfigBox, Box, BoxList

# Assuming these helper functions are available in your project structure
from inferyx.utils.common import get_attribute_by_datapod, get_all_by_type, get_all_versions, get_one_by_uuid_and_version


class AppConfig:
    """Configuration class for application settings."""

    def __init__(self, host: str, appToken: str, adminToken: Optional[str] = None, port: int = 443,
                 verifyHostname: bool = True):
        self.host = host
        self.port = port
        self.appToken = appToken
        self.adminToken = adminToken

    def get_platform_url(self) -> str:
        """Return the platform URL."""
        return f"https://{self.host}/framework"


class DataQuality:
    """
    SDK class to manage Data Quality entities.
    This class handles the creation, retrieval, update, and deletion of a data quality configuration
    via API calls.
    """

    def __init__(self, app_config: AppConfig, **config_dict):
        """
        Initializes the DataQuality object with mandatory and optional parameters.
        This constructor is now more flexible and accepts a dictionary of configuration data.
        """
        if not isinstance(app_config, AppConfig):
            raise TypeError("app_config must be an instance of AppConfig.")

        self.app_config = app_config
        self._config_dict = {
            "name": None,
            "type": "ATTRIBUTE",
            "sourceType": None,
            "sourceName": None,
            "attribute": None,
            "rowKeyType": "DEFAULT",
            "displayName": None,
            "duplicateKeyCheck": "N",
            "enableFilter": "Y",
            "refIntegrityCheck": "Y",
            "active": "Y",
            "locked": "N",
            "published": "N",
            "publicFlag": "N",
            "nullCheck": "N",
            "dataTypeCheck": "",
            "dateFormatCheck": "",
            "customFormatCheck": "",
            "createdDuration": 45795,
            "desc": None,
            "uuid": None,
            "customMessage": {"ref": {"type": "simple"}},
            "dupKeyAttr": [],
            "rowKeyList": [],
            "tags": [],
            "dependsOn": {},
            "blankSpaceCheck": None,
            "valueCheck": [],
            "rangeCheck": {"lowerBound": "", "upperBound": ""},
            "lengthCheck": {"minLength": "", "maxLength": ""},
            "paramList": None,
            "filterInfo": None,
            "thresholdInfo": {"type": "PERCENTAGE", "low": 25, "medium": 50, "high": 75},
            "caseCheck": None,
            "domainCheck": [],
            "refIntegrityAttr": [],
            "filterGroupInfo": []
        }

        # Update the default dictionary with values from the passed config_dict
        self._config_dict.update(config_dict)
        self.config = ConfigBox(self._config_dict)
        self.name = self.config.name

    def create(self) -> dict:
        # This function is unchanged from our last successful code run,
        # where we confirmed the POST request is being sent correctly.
        try:
            print(f"Fetching attribute details for source: '{self.config.sourceName}'...")
            datapod_attributes = get_attribute_by_datapod(self.config.sourceName, self.app_config)
            if not isinstance(datapod_attributes, list) or not datapod_attributes:
                raise ValueError("Datapod exists but has no attributes or the returned format is incorrect.")

            selected_attribute = next(
                (
                    attr
                    for attr in datapod_attributes
                    if attr.get("attrName") == self.config.attribute
                ),
                None
            )

            if not selected_attribute:
                raise ValueError(
                    f"Attribute '{self.config.attribute}' not found in datapod '{self.config.sourceName}'.")

            datapod_uuid = selected_attribute.get("ref").get("uuid")
            attribute_id = selected_attribute.get("attrId")
            attribute_type = selected_attribute.get("attrType")

            print(f"Found datapod UUID: {datapod_uuid}")

        except Exception as e:
            msg = f"❌ Failed to retrieve Datapod details from '{self.config.sourceName}': {e}"
            print(msg)
            raise Exception(msg)

        payload_data = {
            "name": self._config_dict["name"],
            "displayName": self._config_dict["displayName"],
            "type": self._config_dict["type"],
            "duplicateKeyCheck": self._config_dict["duplicateKeyCheck"],
            "enableFilter": self._config_dict["enableFilter"],
            "customMessage": self._config_dict["customMessage"],
            "refIntegrityCheck": self._config_dict["refIntegrityCheck"],
            "dupKeyAttr": self._config_dict["dupKeyAttr"],
            "rowKeyType": self._config_dict["rowKeyType"],
            "rowKeyList": self._config_dict["rowKeyList"],
            "tags": self._config_dict["tags"],
            "active": self._config_dict["active"],
            "locked": self._config_dict["locked"],
            "published": self._config_dict["published"],
            "publicFlag": self._config_dict["publicFlag"],
            "dependsOn": {
                "ref": {
                    "type": "datapod",
                    "uuid": datapod_uuid
                }
            },
            "attribute": {
                "ref": {
                    "type": "datapod",
                    "uuid": datapod_uuid
                },
                "attrId": attribute_id
            },
            "blankSpaceCheck": self._config_dict["blankSpaceCheck"],
            "nullCheck": self._config_dict["nullCheck"],
            "valueCheck": self._config_dict["valueCheck"],
            "rangeCheck": self._config_dict["rangeCheck"],
            "dataTypeCheck": self._config_dict["dataTypeCheck"],
            "dateFormatCheck": self._config_dict["dateFormatCheck"],
            "customFormatCheck": self._config_dict["customFormatCheck"],
            "lengthCheck": self._config_dict["lengthCheck"],
            "paramList": self._config_dict["paramList"],
            "businessDateAttr": {
                "ref": {
                    "uuid": datapod_uuid,
                    "type": "datapod"
                },
                "attrId": attribute_id,
                "attrType": attribute_type
            },
            "filterInfo": self._config_dict["filterInfo"],
            "thresholdInfo": self._config_dict["thresholdInfo"],
            "caseCheck": self._config_dict["caseCheck"],
            "domainCheck": self._config_dict["domainCheck"],
            "refIntegrityAttr": self._config_dict["refIntegrityAttr"],
            "createdDuration": self._config_dict["createdDuration"],
            "filterGroupInfo": self._config_dict["filterGroupInfo"]
        }

        url = f"{self.app_config.get_platform_url()}/common/submit?action=add&type=dq&upd_tag=N"
        headers = {"token": self.app_config.appToken}

        print("Making API call with headers:", headers)
        try:
            print("Making API call with payload:", json.dumps(payload_data, indent=2))
            response = requests.post(url, headers=headers, json=payload_data, verify=False)
            response.raise_for_status()

            print(f"Request Method Sent: {response.request.method}")
            print(f"Request URL Sent: {response.request.url}")

            print("\n--- Request Headers Sent ---")
            for header, value in response.request.headers.items():
                print(f"{header}: {value}")
            print("--- End Request Headers ---\n")

            print("\n--- Request Body Sent ---")
            print(response.request.body)
            print("--- End Request Body ---\n")

            print("\n--- Raw Response Content ---")
            print(response.text)
            print("--- End Raw Response Content ---\n")

            try:
                response_data = response.json()
                if response_data.get("status") == "SUCCESS":
                    print(f"✅ Data Quality configuration '{self.name}' created successfully")
                    return {"status": "SUCCESS", "name": self.name, "response": response_data}
                else:
                    message = response_data.get("message", "Unknown error in response body")
                    raise Exception(f"❌ Failed to create data quality check '{self.name}': {message}")
            except json.JSONDecodeError:
                # Handle cases where the response is not valid JSON
                if response.status_code == 200 and response.text:
                    # Assuming a non-JSON string response indicates success with an ID
                    print(
                        f"✅ Data Quality configuration '{self.name}' created successfully (Non-JSON response received)")
                    return {"status": "SUCCESS", "name": self.name, "response": response.text}
                else:
                    response_data = {"status": "ERROR", "message": "Failed to decode JSON from response."}
                    message = response_data.get("message", "Unknown error in response body")
                    raise Exception(f"❌ Failed to create data quality check '{self.name}': {message}")


        except requests.RequestException as req_err:
            msg = f"❌ Request failed while creating data quality '{self.name}': {req_err}"
            print(msg)
            raise Exception(msg)
        except Exception as e:
            print(f"An error occurred: {e}")
            raise

    def update(self) -> dict:
        """
        Updates an existing data quality configuration.
        """
        print(f"Updating Data Quality configuration '{self.name}'...")
        # Get the UUID of the existing Data Quality configuration
        dq_info = self.get_info()
        if not dq_info or "uuid" not in dq_info:
            raise ValueError(f"Data Quality configuration '{self.name}' not found or has no UUID to update.")

        self.config.uuid = dq_info["uuid"]
        self.config.updatedAt = dq_info["updatedAt"]

        payload_data = self.config.to_dict()

        url = f"{self.app_config.get_platform_url()}/common/submit?action=update&type=dq&upd_tag=N"
        headers = {"token": self.app_config.appToken}

        print("Making API call to update with payload:", json.dumps(payload_data, indent=2))
        try:
            response = requests.post(url, headers=headers, json=payload_data, verify=False)
            response.raise_for_status()

            response_data = response.json()
            if response_data.get("status") == "SUCCESS":
                print(f"✅ Data Quality configuration '{self.name}' updated successfully")
                return {"status": "SUCCESS", "name": self.name, "response": response_data}
            else:
                message = response_data.get("message", "Unknown error in response body")
                raise Exception(f"❌ Failed to update data quality check '{self.name}': {message}")

        except requests.RequestException as req_err:
            msg = f"❌ Request failed while updating data quality '{self.name}': {req_err}"
            print(msg)
            raise Exception(msg)

    def delete(self) -> dict:
        """
        Deletes an existing data quality configuration by its UUID.
        """
        dq_info = self.get_info()
        if not dq_info or "uuid" not in dq_info:
            raise ValueError(f"Data Quality configuration '{self.name}' not found or has no UUID to delete.")

        uuid_to_delete = dq_info["uuid"]
        url = f"{self.app_config.get_platform_url()}/common/submit?action=delete&type=dq&uuid={uuid_to_delete}"
        headers = {"token": self.app_config.appToken}

        print(f"Deleting Data Quality configuration with UUID: {uuid_to_delete}")
        try:
            response = requests.post(url, headers=headers, verify=False)
            response.raise_for_status()

            response_data = response.json()
            if response_data.get("status") == "SUCCESS":
                print(f"✅ Data Quality configuration '{self.name}' deleted successfully")
                return {"status": "SUCCESS", "name": self.name, "response": response_data}
            else:
                message = response_data.get("message", "Unknown error in response body")
                raise Exception(f"❌ Failed to delete data quality check '{self.name}': {message}")

        except requests.RequestException as req_err:
            msg = f"❌ Request failed while deleting data quality '{self.name}': {req_err}"
            print(msg)
            raise Exception(msg)

    def get_info(self) -> dict:
        """
        Retrieves the full configuration details for a single data quality configuration.
        This is an internal helper method used by other functions.
        """
        print(f"Fetching details for Data Quality configuration '{self.name}'...")
        try:
            # Reusing the existing common function to get a list of all DQs
            all_dq = get_all_by_type("dq", self.app_config)

            # Find the configuration by name
            for dq_config in all_dq:
                if dq_config.get("name") == self.name:
                    print(f"✅ Found details for Data Quality configuration '{self.name}'")
                    return dq_config

            print(f"❌ Data Quality configuration '{self.name}' not found.")
            return {}

        except Exception as e:
            msg = f"❌ Failed to retrieve Data Quality details: {e}"
            print(msg)
            return {}

    def get(self, version: Optional[str] = None) -> "DataQuality":
        """
        Retrieves a specific data quality configuration by version or the latest version if none is specified.
        This is the main user-facing method that returns a DataQuality object.
        """
        data = None

        if version:
            versions_list = get_all_versions(type="dq", name=self.name, app_config=self.app_config)
            for item in versions_list:
                if item.get('version') == version:
                    data = get_one_by_uuid_and_version(item.get('uuid'), item.get('version'), 'dq',
                                                       self.app_config)
                    break
            if not data:
                raise ValueError(f"Data Quality version '{version}' not found")
        else:
            # Use get_info to get the latest version data
            data = self.get_info()

        if not data:
            raise ValueError(f"Data Quality configuration '{self.name}' not found")

        # Return a new instance of the DataQuality class with the fetched data
        return DataQuality(app_config=self.app_config, **data)


# Example usage:
if __name__ == "__main__":
    app_config = AppConfig(
        host="dev.inferyx.com",
        appToken="bM8P1i6Xf7O1NpyCOoWmJrrcELE3JkcGzBbguCzB",
        adminToken="iresTHOb208NrFOuLbdrgNNYuUNHYOrCyeQRrISL"
    )

    try:
        # 1. Create a DataQuality object for a non-existent configuration.
        dq_instance = DataQuality(
            app_config=app_config,
            name="1_DQ",
            type="ATTRIBUTE",
            sourceType="datapod",
            sourceName="customer_white_list",
            attribute="customer_id",
            rowKeyType="DEFAULT",
            desc="This is a data quality check for customer data."
        )

        # 2. Test the new get() method to retrieve the latest DQ
        #dq_latest_info = dq_instance.get()
        #print(f"\nRetrieved latest data quality configuration: {dq_latest_info}")
        #print(f"Name of the retrieved configuration: {dq_latest_info.name}")

       # dq_latest_info = dq_instance.get_info()
        #print(f"\nRetrieved latest data quality configuration: {dq_latest_info}")
        #dq_instance.create()

        # dq_instance.delete()
        # dq_instance.update()


    except Exception as e:
        print(f"An error occurred: {e}")
