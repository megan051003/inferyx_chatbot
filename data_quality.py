import json
import requests
from typing import Optional, Any, Dict, List
from box import ConfigBox, Box, BoxList

# Assuming these helper functions are available in your project structure
from inferyx.utils.common import get_attribute_by_datapod, get_all_by_type
from inferyx.utils.collection import DataQualityCollection

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

    def __init__(
            self,
            app_config: AppConfig,
            name: str,
            type: str,
            sourceType: str,
            sourceName: str,
            attribute: str,
            rowKeyType: str,
            displayName: Optional[str] = None,
            duplicateKeyCheck: str = "N",
            enableFilter: str = "Y",
            refIntegrityCheck: str = "Y",
            active: str = "Y",
            locked: str = "N",
            published: str = "N",
            publicFlag: str = "N",
            nullCheck: str = "N",
            dataTypeCheck: str = "",
            dateFormatCheck: str = "",
            customFormatCheck: str = "",
            createdDuration: int = 45795,
            desc: Optional[str] = None,
            uuid: Optional[str] = None,
            customMessage: Dict[str, Any] = None,
            dupKeyAttr: List[Any] = None,
            rowKeyList: List[Any] = None,
            tags: List[Any] = None,
            dependsOn: Dict[str, Any] = None,
            blankSpaceCheck: Optional[Any] = None,
            valueCheck: List[Any] = None,
            rangeCheck: Dict[str, Any] = None,
            lengthCheck: Dict[str, Any] = None,
            paramList: Optional[Any] = None,
            filterInfo: Optional[Any] = None,
            thresholdInfo: Dict[str, Any] = None,
            caseCheck: Optional[Any] = None,
            domainCheck: List[Any] = None,
            refIntegrityAttr: List[Any] = None,
            filterGroupInfo: List[Any] = None
    ):
        """
        Initializes the DataQuality object with mandatory and optional parameters.
        """
        if not isinstance(app_config, AppConfig):
            raise TypeError("app_config must be an instance of AppConfig.")

        self.app_config = app_config
        self._config_dict = {
            "name": name,
            "type": type,
            "sourceType": sourceType,
            "sourceName": sourceName,
            "attribute": attribute,
            "rowKeyType": rowKeyType,
            "displayName": displayName or name,
            "duplicateKeyCheck": duplicateKeyCheck,
            "enableFilter": enableFilter,
            "refIntegrityCheck": refIntegrityCheck,
            "active": active,
            "locked": locked,
            "published": published,
            "publicFlag": publicFlag,
            "nullCheck": nullCheck,
            "dataTypeCheck": dataTypeCheck,
            "dateFormatCheck": dateFormatCheck,
            "customFormatCheck": customFormatCheck,
            "createdDuration": createdDuration,
            "desc": desc,
            "uuid": uuid,
            "customMessage": customMessage or {"ref": {"type": "simple"}},
            "dupKeyAttr": dupKeyAttr or [],
            "rowKeyList": rowKeyList or [],
            "tags": tags or [],
            "dependsOn": dependsOn or {},
            "blankSpaceCheck": blankSpaceCheck,
            "valueCheck": valueCheck or [],
            "rangeCheck": rangeCheck or {"lowerBound": "", "upperBound": ""},
            "lengthCheck": lengthCheck or {"minLength": "", "maxLength": ""},
            "paramList": paramList,
            "filterInfo": filterInfo,
            "thresholdInfo": thresholdInfo or {"type": "PERCENTAGE", "low": 25, "medium": 50, "high": 75},
            "caseCheck": caseCheck,
            "domainCheck": domainCheck or [],
            "refIntegrityAttr": refIntegrityAttr or [],
            "filterGroupInfo": filterGroupInfo or []
        }
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
            except json.JSONDecodeError:
                response_data = {"status": "ERROR", "message": "Failed to decode JSON from response."}

            if response_data.get("status") == "SUCCESS":
                print(f"✅ Data Quality configuration '{self.name}' created successfully")
                return {"status": "SUCCESS", "name": self.name, "response": response_data}
            else:
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
        """
        print(f"Fetching details for Data Quality configuration '{self.name}'...")
        try:
            dq_info = DataQualityCollection.collection(app_config=self.app_config, name=self.name)
            print(f"✅ Found details for Data Quality configuration '{self.name}'")
            return dq_info

        except Exception as e:
            msg = f"❌ Failed to retrieve Data Quality details for '{self.name}': {e}"
            print(msg)
            return {}

    def get(self) -> BoxList:
        """
        Retrieves a collection of all data quality configurations.
        """
        print("Fetching all Data Quality configurations...")
        try:
            # Reusing the existing common function to get a list of all DQs
            all_dq = get_all_by_type("dq", self.app_config)

            print("✅ All Data Quality configurations retrieved successfully.")
            return all_dq

        except Exception as e:
            msg = f"❌ Failed to retrieve Data Quality configurations: {e}"
            print(msg)
            raise Exception(msg)


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
            name="MyFirstDataQualityCheck_fromSDK",
            type="ATTRIBUTE",
            sourceType="datapod",
            sourceName="customer_white_list",
            attribute="customer_id",
            rowKeyType="DEFAULT",
            desc="This is a data quality check for customer data."
        )

        # 2. Test the new get_info method
        dq_info = dq_instance.get_info()
        print(f"\nget_info result: {dq_info}")

        # 3. Test the create method
        print("\n--- Testing create method ---")
        # dq_instance.create()

        # 4. Test the delete method (requires a valid UUID)
        print("\n--- Testing delete method ---")
        # dq_instance.delete()

        # 5. Test the update method
        print("\n--- Testing update method ---")
        # You would typically change a value here before updating, e.g., dq_instance.config.desc = "New description"
        # dq_instance.update()

        # 6. Test the get() method to retrieve all DQs
        print("\n--- Testing get (all) method ---")
        all_dqs = dq_instance.get()
        print(f"Retrieved {len(all_dqs)} data quality configurations.")


    except Exception as e:
        print(f"An error occurred: {e}")
