from pathlib import Path
import requests

try:
    import urllib.parse as urlparse  # Python 3
except ImportError:
    import urlparse  # Python 2

import argparse
import datetime
import logging
import os
import re
from io import BytesIO
import sys
import time

from zipfile import ZipFile

import yaml

requests.packages.urllib3.disable_warnings() 

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)

import pdb

# The regex to find the measurement id from the measurement page
# This should be read from the uploaded file, but would require an extra NetCDF module.
regex = "<h3>Measurement (?P<measurement_id>.{12,15}) <small>"  # {12, 15} to handle both old- and new-style measurement ids.


class SCC:
    """A simple class that will attempt to upload a file on the SCC server.

    The uploading is done by simulating a normal browser session. In the current
    version no check is performed, and no feedback is given if the upload
    was successful. If everything is setup correctly, it will work.
    """

    def __init__(self, auth, output_dir: Path | str | None, base_url: str):
        self.auth = auth
        self.base_url = base_url
        self.output_dir = self._set_output_dir(output_dir)
        self.session = requests.Session()
        self.session.auth = auth
        self.session.verify = False

        self.login_url: str = urlparse.urljoin(self.base_url, "accounts/login/")
        self.logout_url = urlparse.urljoin(self.base_url, "accounts/logout/")
        self.list_measurements_url = urlparse.urljoin(
            self.base_url, "data_processing/measurements/"
        )

        self.upload_url = urlparse.urljoin(
            self.base_url, "data_processing/measurements/quick/"
        )

        self.download_raw_pattern = urlparse.urljoin(
            self.base_url, "data_processing/measurements/{0}/download-raw/"
        )

        self.download_hirelpp_pattern = urlparse.urljoin(
            self.base_url, "data_processing/measurements/{0}/download-hirelpp/"
        )

        self.download_cloudmask_pattern = urlparse.urljoin(
            self.base_url, "data_processing/measurements/{0}/download-cloudmask/"
        )

        self.download_preprocessed_pattern = urlparse.urljoin(
            self.base_url, "data_processing/measurements/{0}/download-preprocessed/"
        )
        self.download_optical_pattern = urlparse.urljoin(
            self.base_url, "data_processing/measurements/{0}/download-optical/"
        )
        self.download_graph_pattern = urlparse.urljoin(
            self.base_url, "data_processing/measurements/{0}/download-plots/"
        )
        self.download_elic_pattern = urlparse.urljoin(
            self.base_url, "data_processing/measurements/{0}/download-elic/"
        )
        self.download_elquick_pattern = urlparse.urljoin(
            self.base_url, "data_processing/measurements/{0}/download-elquick/"
        )
        self.delete_measurement_pattern = urlparse.urljoin(
            self.base_url, "admin/database/measurements/{0}/delete/"
        )

        self.api_base_url = urlparse.urljoin(self.base_url, "api/v1/")
        self.api_measurement_pattern = urlparse.urljoin(
            self.api_base_url, "measurements/{0}/"
        )
        self.api_measurements_url = urlparse.urljoin(self.api_base_url, "measurements")
        self.api_sounding_search_pattern = urlparse.urljoin(
            self.api_base_url, "sounding_files/?filename={0}"
        )
        self.api_lidarratio_search_pattern = urlparse.urljoin(
            self.api_base_url, "lidarratio_files/?filename={0}"
        )
        self.api_overlap_search_pattern = urlparse.urljoin(
            self.api_base_url, "overlap_files/?filename={0}"
        )
        
    def _set_output_dir(self, output_dir: Path | str | None):
        if output_dir is None:
            _output_dir = Path.cwd()
        elif isinstance(output_dir, str):
            _output_dir = Path(output_dir)
        elif isinstance(output_dir, Path):
            _output_dir = output_dir
        return _output_dir

    def login(self, credentials: list[str]):
        """Login to SCC."""
        logger.debug("Attempting to login to SCC, username %s." % credentials[0])
        login_credentials = {"username": credentials[0], "password": credentials[1]}

        logger.debug("Accessing login page at %s." % self.login_url)

        # Get upload form
        login_page = self.session.get(self.login_url)

        if not login_page.ok:
            raise self.PageNotAccessibleError(
                "Could not access login pages. Status code %s" % login_page.status_code
            )

        logger.debug("Submitting credentials.")
        # Submit the login data
        login_submit = self.session.post(
            self.login_url,
            data=login_credentials,
            headers={
                "X-CSRFToken": login_page.cookies["csrftoken"],
                "referer": self.login_url,
            },
        )
        return login_submit

    def logout(self):
        """Logout from SCC"""
        return self.session.get(self.logout_url, stream=True)

    def upload_file(
        self, filename, system_id, rs_filename=None, ov_filename=None, lr_filename=None
    ):
        """Upload a filename for processing with a specific system. If the
        upload is successful, it returns the measurement id."""
        # Get submit page
        upload_page = self.session.get(self.upload_url)

        # Submit the data
        upload_data = {"system": system_id}
        files = {"data": open(filename, "rb")}
        if rs_filename is not None:
            ancillary_file, _ = self.get_ancillary(rs_filename, "sounding")
            if isinstance(ancillary_file, AncillaryFile):
                if ancillary_file.already_on_scc:
                    logger.warning(
                        "Sounding file {0.filename} already on the SCC with id {0.id}. Ignoring it.".format(
                            ancillary_file
                        )
                    )
                else:
                    logger.debug("Adding sounding file %s" % rs_filename)
                    files["sounding_file"] = open(rs_filename, "rb")

        if ov_filename is not None:
            ancillary_file, _ = self.get_ancillary(ov_filename, "overlap")
            if isinstance(ancillary_file, AncillaryFile):
                if ancillary_file.already_on_scc:
                    logger.warning(
                        "Overlap file {0.filename} already on the SCC with id {0.id}. Ignoring it.".format(
                            ancillary_file
                        )
                    )
                else:
                    logger.debug("Adding overlap file %s" % ov_filename)
                    files["overlap_file"] = open(ov_filename, "rb")

        if lr_filename is not None:
            ancillary_file, _ = self.get_ancillary(lr_filename, "lidarratio")
            if isinstance(ancillary_file, AncillaryFile):
                if ancillary_file.already_on_scc:
                    logger.warning(
                        "Lidar ratio file {0.filename} already on the SCC with id {0.id}. Ignoring it.".format(
                            ancillary_file
                        )
                    )
                else:
                    logger.debug("Adding lidar ratio file %s" % lr_filename)
                    files["lidar_ratio_file"] = open(lr_filename, "rb")

        logger.info("Uploading of file(s) %s started." % filename)

        upload_submit = self.session.post(
            self.upload_url,
            data=upload_data,
            files=files,
            headers={
                "X-CSRFToken": upload_page.cookies["csrftoken"],
                "referer": self.upload_url,
            },
        )

        if upload_submit.status_code != 200:
            logger.warning(
                "Connection error. Status code: %s" % upload_submit.status_code
            )
            return False

        # Check if there was a redirect to a new page.
        if upload_submit.url == self.upload_url:
            measurement_id = False
            logger.error(
                "Uploaded file(s) rejected! Try to upload manually to see the error."
            )
        else:
            measurement_id = re.findall(regex, upload_submit.text)[0]
            logger.info(
                "Successfully uploaded measurement with id %s." % measurement_id
            )

        return measurement_id

    def download_files(self, measurement_id, scc_type, download_url):
        """Downloads some files from the download_url to the specified
        SCC type. This method is used to download preprocessed file, optical
        files etc.
        """
        # TODO: Make downloading more robust (e.g. in case that files do not exist on server).
        # Get the file
        request = self.session.get(download_url, stream=True)

        if not request.ok:
            raise Exception(
                "Could not download files for measurement '%s'" % measurement_id
            )

        # Create the dir if it does not exist
        local_dir = self.output_dir
        local_dir.mkdir(parents=True, exist_ok=True)

        # Get the file name from the URL or provide a custom name
        scc_product = download_url.split("/")[-2].split("-")[-1]
        file_name = f"{scc_product}_{measurement_id}.zip"

        # Specify the local path for saving the file
        local_file_path = local_dir / file_name
        # Save the file by chunk, needed if the file is big
        with local_file_path.open("wb") as local_file:
            for chunk in request.iter_content(chunk_size=1024):
                if chunk:
                    local_file.write(chunk)
        logger.info(f"{scc_type} files downloaded.")

    def download_raw(self, measurement_id):
        """Download raw files for the measurement id."""
        # Construct the download url
        download_url = self.download_raw_pattern.format(measurement_id)
        self.download_files(measurement_id, "raw", download_url)

    def download_hirelpp(self, measurement_id):
        """Download HiRElPP files for the measurement id."""
        # Construct the download url
        download_url = self.download_hirelpp_pattern.format(measurement_id)
        self.download_files(measurement_id, "hirelpp", download_url)

    def download_cloudmask(self, measurement_id):
        """Download preprocessed files for the measurement id."""
        # Construct the download url
        download_url = self.download_cloudmask_pattern.format(measurement_id)
        self.download_files(measurement_id, "cloudmask", download_url)

    def download_preprocessed(self, measurement_id):
        """Download preprocessed files for the measurement id."""
        # Construct the download url
        download_url = self.download_preprocessed_pattern.format(measurement_id)
        self.download_files(measurement_id, "scc_preprocessed", download_url)

    def download_optical(self, measurement_id):
        """Download optical files for the measurement id."""
        # Construct the download url
        download_url = self.download_optical_pattern.format(measurement_id)
        self.download_files(measurement_id, "scc_optical", download_url)

    def download_graphs(self, measurement_id):
        """Download profile graphs for the measurement id."""
        # Construct the download url
        download_url = self.download_graph_pattern.format(measurement_id)
        self.download_files(measurement_id, "scc_plots", download_url)

    def download_elic(self, measurement_id):
        """Download profile graphs for the measurement id."""
        # Construct the download url
        download_url = self.download_elic_pattern.format(measurement_id)
        self.download_files(measurement_id, "elic", download_url)

    def download_elquick(self, measurement_id):
        """Download elquick for the measurement id."""
        # Construct the download url
        download_url = self.download_elquick_pattern.format(measurement_id)
        # self.download_files(measurement_id, 'elquick', download_url)
        logger.warning("Elquick download not available yet")

    def rerun_processing(self, measurement_id, monitor=True):
        measurement, status = self.get_measurement(measurement_id)

        if measurement:
            request = self.session.get(measurement.rerun_processing_url, stream=True)

            if request.status_code != 200:
                logger.error(
                    "Could not rerun processing for %s. Status code: %s"
                    % (measurement_id, request.status_code)
                )
                return

            if monitor:
                self.monitor_processing(measurement_id)

    def rerun_all(self, measurement_id, monitor=True):
        logger.debug("Started rerun_all procedure.")

        logger.debug("Getting measurement %s" % measurement_id)
        measurement, status = self.get_measurement(measurement_id)

        if measurement:
            logger.debug(
                "Attempting to rerun all processing through %s."
                % measurement.rerun_all_url
            )

            request = self.session.get(measurement.rerun_all_url, stream=True)

            if request.status_code != 200:
                logger.error(
                    "Could not rerun pre processing for %s. Status code: %s"
                    % (measurement_id, request.status_code)
                )
                return

            if monitor:
                self.monitor_processing(measurement_id)

    def process(
        self,
        filename,
        system_id,
        monitor,
        rs_filename=None,
        lr_filename=None,
        ov_filename=None,
    ):
        """Upload a file for processing and wait for the processing to finish.
        If the processing is successful, it will download all produced files.
        """
        logger.info("--- Processing started on %s. ---" % datetime.datetime.now())
        # Upload file
        logger.info("--- Uploading file")
        measurement_id = self.upload_file(
            filename,
            system_id,
            rs_filename=rs_filename,
            lr_filename=lr_filename,
            ov_filename=ov_filename,
        )

        if measurement_id and monitor:
            logger.info("--- Monitoring processing")
            return self.monitor_processing(measurement_id)

        return None

    def monitor_processing(
        self,
        measurement_id: str,
        download_raw: bool = False,
        download_hirelpp: bool = False,
        download_cloudmask: bool = False,
        download_elpp: bool =True,
        download_elda: bool =True,
        download_elic: bool = False,
        download_elquick: bool = False,
        download_eldec: bool = False,
        upload_max: int = 10,
        time_sleep_upload: int = 10,
        queue_max: int = 10,
        time_sleep_queue: int = 10,
        run_max: int = 20,
        time_sleep_run: int = 10,        
    ):
        """Monitor the processing progress of a measurement id. If the download flags are set to True, the files will be downloaded when the processing is finished.

        Args:
            measurement_id (str): Measurement id.
            download_raw (bool, optional): raw files will be downloared. Defaults to False.
            download_hirelpp (bool, optional): hirelpp files will be downloared. Defaults to False.
            download_cloudmask (bool, optional): cloudmask files will be downloared. Defaults to False.
            download_elpp (bool, optional): elpp files will be downloared. Defaults to True.
            download_elda (bool, optional): elda files will be downloared. Defaults to True.
            download_elic (bool, optional): elic files will be downloared. Defaults to False.
            download_elquick (bool, optional): elquick files will be downloared. Defaults to False.
            download_eldec (bool, optional): elded files will be downloared. Defaults to False.
            upload_max (int, optional): Maximum number of times of `time_sleep_upload` will wait file to be uploaded to the SCC. Default 10.
            time_sleep_upload (int, optional): Time in seconds to wait file to be uploaded to the SCC. Default 10 seconds. 
            queue_max (int, optional): Maximum number of times of `time_sleep_queue` will wait file to leave queue. Default 10.
            time_sleep_queue (int, optional): Time in seconds to wait file to leave queue to the SCC. Default 10 seconds. 
            run_max (int, optional): Maximum number of times of `time_sleep_run` will wait file to finish processing. Default 10.
            time_sleep_run (int, optional): Time in seconds to wait file to finish processing. Default 10 seconds. 

        Raises:
            TimeoutError: File not found in the SCC DB.
            Warning: Measurement is delayed.
        """
        upload_count, queue_count, run_count = 0, 0, 0

        # try to wait for measurement to appear in API
        measurement = None
        logger.info("Looking for measurement %s in SCC", measurement_id)
        while upload_count < upload_max:
            time.sleep(time_sleep_upload)
            measurement, status = self.get_measurement(measurement_id)
            if status == 200:
                logger.info(f"Measurement found: {measurement_id}")
                break
            else:
                if upload_count < upload_max:
                    logger.error("Measurement not found yet. waiting %ds", time_sleep_upload)
                    upload_count += 1
                else:
                    raise TimeoutError(f"Measurement {measurement_id} doesn't seem to exist.")

        if measurement is None:
            raise FileNotFoundError(f"Measurement {measurement_id} doesn't seem to exist.")
            
        # Start download of hirelpp files
        products = [
            "upload",
            "hirelpp",
            "cloudmask",
            "elpp",
            "elda",
            "elquick",
            "elic",
            "eldec",
        ]
        measurement, status = self.get_measurement(measurement_id)

        if measurement is None:
            raise FileNotFoundError(f"Measurement {measurement_id} doesn't seem to exist.")

        status = [
            measurement.upload, #raw 
            measurement.hirelpp,
            measurement.cloudmask,
            measurement.elpp,
            measurement.elda,
            measurement.elquick,
            measurement.elic,
            measurement.eldec,
        ]
        status_dict = dict(zip(products, status))

        required_downloads = [
            download_raw,
            download_hirelpp,
            download_cloudmask,
            download_elpp,
            download_elda,
            download_elquick,
            download_elic,
            download_eldec,
        ]
        required_products = [
            product
            for product, required in zip(products, required_downloads)
            if required
        ]
        
        if self.is_being_processed(measurement_id):
            logger.info(f"Measurement {measurement_id} is being processed")
        elif self.is_queued(measurement_id):
            logger.info(f"Measurement {measurement_id} is queued")
        elif self.is_delayed(measurement_id):
            raise Warning(f"Measurement {measurement_id} is delayed")


        # Wait for processing to finish
        if self.is_queued(measurement_id):
            while (queue_count <= queue_max):
                logger.info(f"#{queue_count:02d}/{queue_max}: Process is queued. Please wait. ")
                time.sleep(time_sleep_queue)
                measurement, status = self.get_measurement(measurement_id)
                if measurement is None:
                    raise FileNotFoundError(f"Measurement {measurement_id} doesn't seem to exist.")
                                    
                if not measurement.is_queued:
                    logger.info("Process is running.")
                    break
                else:
                    queue_count += 1
            
            if queue_count > queue_max:
                raise TimeoutError('Process was queued too long.')

        if self.is_being_processed(measurement_id):
            while (run_count <= run_max):
                info_str = f"RAW: {measurement.upload}, HIRELPP: {measurement.hirelpp}, CLOUDMASK: {measurement.cloudmask}, ELPP: {measurement.elpp}, ELDA: {measurement.elda}, ELIC: {measurement.elic}, ELQUICK: {measurement.elquick}"
                logger.info(f"#{run_count:02d}/{run_max}: Please wait. " + info_str)
                
                time.sleep(time_sleep_run)
                
                measurement, status = self.get_measurement(measurement_id)
                
                if measurement is None:
                    raise FileNotFoundError(f"Measurement {measurement_id} doesn't seem to exist.")            
                
                if not self.is_being_processed(measurement_id):
                    logger.info("--- SCC processing finished. ---")
                    break

                if all([getattr(measurement, product)==127 for product in required_products]):
                    logger.info(f"All SCC products to be downloaded have been already processed by the SCC.")
                    break
                elif getattr(measurement, "elpp") == -127:
                    logger.warning(f"ELPP error code {measurement.elpp_exit_code['exit_code']}: {measurement.elpp_exit_code['description']}")
                    break
                run_count += 1
            
            if run_count == run_max:
                logger.warning(f"Timeout reached. Not all products were processed.")
                return measurement

        for product in required_products:            
            if status_dict[product] == 127:
                if product == "upload":
                    self.download_files(
                        measurement_id,
                        product,
                        self.download_raw_pattern.format(measurement_id),
                    )
                if product == "hirelpp":
                    self.download_files(
                        measurement_id,
                        product,
                        self.download_hirelpp_pattern.format(measurement_id),
                    )
                elif product == "cloudmask":
                    self.download_files(
                        measurement_id,
                        product,
                        self.download_cloudmask_pattern.format(measurement_id),
                    )
                elif product == "elpp":
                    self.download_files(
                        measurement_id,
                        product,
                        self.download_preprocessed_pattern.format(measurement_id),
                    )
                elif product == "elda":
                    self.download_files(
                        measurement_id,
                        product,
                        self.download_optical_pattern.format(measurement_id),
                    )
                elif product == "elquick":
                    self.download_files(
                        measurement_id,
                        product,
                        self.download_elquick_pattern.format(measurement_id),
                    )
                elif product == "elic":
                    self.download_files(
                        measurement_id,
                        product,
                        self.download_elic_pattern.format(measurement_id),
                    )
                elif product == "eldec":
                    logger.info(f"ELDEC downloading not implemented yet.")
                    # self.download_files(measurement_id, product, self.download_eldec_pattern.format(measurement_id))
            elif status_dict[product] == -127:
                logger.warning(f"{product} not avaiable in the SCC.")
                if product == "hirelpp":
                    logger.warning(
                        f"{product} error code {measurement.hirelpp_exit_code['exit_code']}: {measurement.hirelpp_exit_code['description']}"
                    )
                elif product == "cloudmask":
                    logger.warning(
                        f"{product} error code {measurement.cloudmask_exit_code['exit_code']}: {measurement.cloudmask_exit_code['description']}"
                    )
                elif product == "elpp":
                    logger.warning(
                        f"{product} error code {measurement.elpp_exit_code['exit_code']}: {measurement.elpp_exit_code['description']}"
                    )
                elif product == "elda":
                    logger.warning(
                        f"{product} error code {measurement.elda_exit_code['exit_code']}: {measurement.elda_exit_code['description']}"
                    )
                elif product == "elquick":
                    logger.warning(
                        f"{product} error code {measurement.elquick_exit_code['exit_code']}: {measurement.elquick_exit_code['description']}"
                    )
                elif product == "elic":
                    logger.warning(
                        f"{product} error code {measurement.elic_exit_code['exit_code']}: {measurement.elic_exit_code['description']}"
                    )
                elif product == "eldec":
                    logger.warning(
                        f"{product} error code {measurement.eldec_exit_code['exit_code']}: {measurement.eldec_exit_code['description']}"
                    )
                else:
                    logger.warning(f"Unknown error code for {product}: {status_dict[product]}")

    def get_measurement(self, measurement_id):
        measurement_url = self.api_measurement_pattern.format(measurement_id)
        logger.debug("Measurement API URL: %s" % measurement_url)

        response = self.session.get(measurement_url)
        if not response.ok:
            logger.error("Could not access API. Status code %s." % response.status_code)
            return None, response.status_code

        response_dict = response.json()
    
        if response_dict:
            measurement = Measurement(self.base_url, response_dict)
            return measurement, response.status_code
        else:
            logger.error("No measurement with id %s found on the SCC." % measurement_id)
            return None, response.status_code

    def is_being_processed(self, measurement_id):
        measurement_url = self.api_measurement_pattern.format(measurement_id)
        logger.debug("Measurement API URL: %s" % measurement_url)

        response = self.session.get(measurement_url)
        if not response.ok:
            logger.error("Could not access API. Status code %s." % response.status_code)
            return None, response.status_code
        else:
            response_dict = response.json()
            return response_dict["is_being_processed"]

    def is_delayed(self, measurement_id):
        measurement_url = self.api_measurement_pattern.format(measurement_id)
        logger.debug("Measurement API URL: %s" % measurement_url)

        response = self.session.get(measurement_url)
        if not response.ok:
            logger.error("Could not access API. Status code %s." % response.status_code)
            return None, response.status_code
        else:
            response_dict = response.json()
            return response_dict["is_delayed"]

    def is_queued(self, measurement_id):
        measurement_url = self.api_measurement_pattern.format(measurement_id)
        logger.debug("Measurement API URL: %s" % measurement_url)

        response = self.session.get(measurement_url)
        if not response.ok:
            logger.error("Could not access API. Status code %s." % response.status_code)
            return None, response.status_code
        else:
            response_dict = response.json()
            return response_dict["is_queued"]

    def delete_measurement(self, measurement_id: str):
        """Deletes a measurement with the provided measurement id. The user
        should have the appropriate permissions.

        The procedures is performed directly through the web interface and
        NOT through the API.
        """
        # Get the measurement object
        measurement, _ = self.get_measurement(measurement_id)

        # Check that it exists
        if measurement is None:
            logger.warning("Nothing to delete.")
            return None

        # Go the the page confirming the deletion

        delete_url = self.delete_measurement_pattern.format(measurement_id)

        confirm_page = self.session.get(delete_url)
        # Check that the page opened properly
        if confirm_page.status_code != 200:
            logger.warning(
                "Could not open delete page. Status: {0}".format(
                    confirm_page.status_code
                )
            )
            return None

        # Delete the measurement
        delete_page = self.session.post(
            delete_url,
            data={"post": "yes"},
            headers={
                "X-CSRFToken": confirm_page.cookies["csrftoken"],
                "referer": delete_url,
            },
        )

        if not delete_page.ok:
            logger.warning(
                "Something went wrong. Delete page status: {0}".format(
                    delete_page.status_code
                )
            )
            return None

        logger.info("Deleted measurement {0}".format(measurement_id))
        return True

    def available_measurements(self):
        """Get a list of available measurement on the SCC."""

        response = self.session.get(self.api_measurements_url)
        response_dict = response.json()

        if response_dict:
            measurement_list = response_dict["objects"]
            measurements = [
                Measurement(self.base_url, measurement_dict)
                for measurement_dict in measurement_list
            ]
            logger.info("Found %s measurements on the SCC." % len(measurements))
        else:
            logger.warning(
                "No response received from the SCC when asked for available measurements."
            )
            measurements = None

        return measurements

    def list_measurements(
        self,
        station=None,
        system=None,
        start=None,
        stop=None,
        upload_status=None,
        processing_status=None,
        optical_processing=None,
    ):
        # TODO: Change this to work through the API

        # Need to set to empty string if not specified, we won't get any results
        params = {
            "station": station if station is not None else "",
            "system": system if system is not None else "",
            "stop": stop if stop is not None else "",
            "start": start if start is not None else "",
            "upload_status": upload_status if upload_status is not None else "",
            "preprocessing_status": processing_status
            if processing_status is not None
            else "",
            "optical_processing_status": optical_processing
            if optical_processing is not None
            else "",
        }

        response_txt = self.session.get(self.list_measurements_url, params=params).text
        tbl_rgx = re.compile(r'<table id="measurements">(.*?)</table>', re.DOTALL)
        entry_rgx = re.compile(r"<tr>(.*?)</tr>", re.DOTALL)
        measurement_rgx = re.compile(
            r'.*?<td><a[^>]*>(\w+)</a>.*?<td>.*?<td>([\w-]+ [\w:]+)</td>.*<td data-order="([-]?\d+),([-]?\d+),([-]?\d+)".*',
            re.DOTALL,
        )
        matches = tbl_rgx.findall(response_txt)
        if len(matches) != 1:
            return []

        ret = []
        for entry in entry_rgx.finditer(matches[0]):
            m = measurement_rgx.match(entry.string[entry.start(0) : entry.end(0)])
            if m:
                name, date, upload, preproc, optical = m.groups()
                ret.append(
                    Measurement(
                        self.base_url,
                        {
                            "id": name,
                            "upload": int(upload),
                            "pre_processing": int(preproc),
                            "processing": int(optical),
                        },
                    )
                )

        return ret

    def measurement_id_for_date(self, t1, call_sign, base_number=0):
        """Give the first available measurement id on the SCC for the specific
        date.
        """
        date_str = t1.strftime("%Y%m%d")
        base_id = "%s%s" % (date_str, call_sign)
        search_url = urlparse.urljoin(
            self.api_base_url, "measurements/?id__startswith=%s" % base_id
        )

        response = self.session.get(search_url)

        response_dict = response.json()

        measurement_id = None

        if response_dict:
            measurement_list = response_dict["objects"]

            if len(measurement_list) == 100:
                raise ValueError("No available measurement id found.")

            existing_ids = [
                measurement_dict["id"] for measurement_dict in measurement_list
            ]

            measurement_number = base_number
            measurement_id = "%s%02i" % (base_id, measurement_number)

            while measurement_id in existing_ids:
                measurement_number = measurement_number + 1
                measurement_id = "%s%02i" % (base_id, measurement_number)

        return measurement_id

    def get_ancillary(self, file_path, file_type):
        """
        Try to get the ancillary file data from the SCC API.

        The result will always be an API object. If the file does not exist, the .exists property is set to False.

        Parameters
        ----------
        file_path : str
           Path  of the uploaded file.
        file_type : str
           Type of ancillary file. One of 'sounding', 'overlap', 'lidarratio'.

        Returns
        : AncillaryFile
           The api object.
        """
        assert file_type in ["sounding", "overlap", "lidarratio"]

        filename = os.path.basename(file_path)

        if file_type == "sounding":
            file_url = self.api_sounding_search_pattern.format(filename)
        elif file_type == "overlap":
            file_url = self.api_overlap_search_pattern.format(filename)
        else:
            file_url = self.api_lidarratio_search_pattern.format(filename)

        response = self.session.get(file_url)

        if not response.ok:
            logger.error("Could not access API. Status code %s." % response.status_code)
            return None, response.status_code

        response_dict = response.json()
        object_list = response_dict["objects"]

        logger.debug("Ancillary file JSON: {0}".format(object_list))

        if object_list:
            ancillary_file = AncillaryFile(
                self.api_base_url, response_dict #TODO:Check if response_dict should be object_list
            )  # Assume only one file is returned
        else:
            ancillary_file = AncillaryFile(
                self.api_base_url, None
            )  # Create an empty object

        return ancillary_file, response.status_code

    class PageNotAccessibleError(RuntimeError):
        pass


class ApiObject(object):
    """A generic class object."""

    def __init__(self, base_url, dict_response):
        self.base_url = base_url

        if dict_response:
            # Add the dictionary key value pairs as object properties
            for key, value in dict_response.items():
                # logger.debug('Setting key {0} to value {1}'.format(key, value))
                try:
                    setattr(self, key, value)
                except:
                    logger.warning(
                        "Could not set attribute {0} to value {1}".format(key, value)
                    )
            self.exists = True
        else:
            self.exists = False


class Measurement(ApiObject):
    """This class represents the measurement object as returned in the SCC API."""
    def __init__(self, base_url, dict_response):
        super(Measurement, self).__init__(base_url, dict_response)
        self.id = self.id
        self.upload = self.upload 
        self.hirelpp = self.hirelpp
        self.cloudmask = self.cloudmask
        self.elpp = self.elpp
        self.elda = self.elda
        self.elquick = self.elquick
        self.elic = self.elic
        self.eldec = self.eldec
        self.is_being_processed: bool = self.is_being_processed
        self.is_delayed: bool = self.is_delayed
        self.is_queued: bool = self.is_queued
        self.pre_processing: bool = False
        self.processing: bool = False
        self.elpp_exit_code: dict = self.elpp_exit_code
        self.elda_exit_code: dict = self.elda_exit_code
        self.elquick_exit_code: dict = self.elquick_exit_code
        self.elic_exit_code: dict = self.elic_exit_code
        self.eldec_exit_code: dict = self.eldec_exit_code
        self.hirelpp_exit_code: dict = self.hirelpp_exit_code
        self.cloudmask_exit_code: dict = self.cloudmask_exit_code  


    @property
    def rerun_processing_url(self):
        url_pattern = urlparse.urljoin(
            self.base_url, "data_processing/measurements/{0}/rerun-elda/"
        )
        return url_pattern.format(self.id) #type: ignore

    @property
    def rerun_all_url(self):
        ulr_pattern = urlparse.urljoin(
            self.base_url, "data_processing/measurements/{0}/rerun-all/"
        )
        return ulr_pattern.format(self.id) #type: ignore

    def __str__(self):
        return "%s: %s, %s, %s" % (
            self.id, #type: ignore
            self.upload, #type: ignore
            self.pre_processing,
            self.processing,
        )


class AncillaryFile(ApiObject):
    """This class represents the ancilalry file object as returned in the SCC API."""
    def __init__(self, base_url, dict_response):
        super(AncillaryFile, self).__init__(base_url, dict_response)

    @property
    def already_on_scc(self):        
        return self.exists

    def __str__(self):
        return "%s: %s, %s" % (self.id, self.filename, self.status) #type: ignore


def process_file(
    filename,
    system_id,
    settings,
    monitor=True,
    rs_filename=None,
    lr_filename=None,
    ov_filename=None,
):
    """Shortcut function to process a file to the SCC."""
    logger.info("Processing file %s, using system %s" % (filename, system_id))

    scc = SCC(
        settings["basic_credentials"], settings["output_dir"], settings["base_url"]
    )
    scc.login(settings["website_credentials"])
    measurement = scc.process(
        filename,
        system_id,
        monitor=monitor,
        rs_filename=rs_filename,
        lr_filename=lr_filename,
        ov_filename=ov_filename,
    )
    scc.logout()
    return measurement


def delete_measurements(measurement_ids, settings):
    """Shortcut function to delete measurements from the SCC."""
    scc = SCC(
        settings["basic_credentials"], settings["output_dir"], settings["base_url"]
    )
    scc.login(settings["website_credentials"])
    for m_id in measurement_ids:
        logger.info("Deleting %s" % m_id)
        scc.delete_measurement(m_id)
    scc.logout()


def rerun_all(measurement_ids, monitor, settings):
    """Shortcut function to rerun measurements from the SCC."""

    scc = SCC(
        settings["basic_credentials"], settings["output_dir"], settings["base_url"]
    )
    scc.login(settings["website_credentials"])
    for m_id in measurement_ids:
        logger.info("Rerunning all products for %s" % m_id)
        scc.rerun_all(m_id, monitor)
    scc.logout()


def rerun_processing(measurement_ids, monitor, settings):
    """Shortcut function to delete a measurement from the SCC."""

    scc = SCC(
        settings["basic_credentials"], settings["output_dir"], settings["base_url"]
    )
    scc.login(settings["website_credentials"])
    for m_id in measurement_ids:
        logger.info("Rerunning (optical) processing for %s" % m_id)
        scc.rerun_processing(m_id, monitor)
    scc.logout()


def list_measurements(
    settings,
    station=None,
    system=None,
    start=None,
    stop=None,
    upload_status=None,
    preprocessing_status=None,
    optical_processing=None,
):
    """List all available measurements"""
    scc = SCC(
        settings["basic_credentials"], settings["output_dir"], settings["base_url"]
    )
    scc.login(settings["website_credentials"])
    ret = scc.list_measurements(
        station=station,
        system=system,
        start=start,
        stop=stop,
        upload_status=upload_status,
        processing_status=preprocessing_status,
        optical_processing=optical_processing,
    )
    for entry in ret:
        print("%s" % entry.id)
    scc.logout()


def download_measurements(
    measurement_ids, download_preproc, download_optical, download_graph, settings
):
    """Download all measurements for the specified IDs"""
    scc = SCC(
        settings["basic_credentials"], settings["output_dir"], settings["base_url"]
    )
    scc.login(settings["website_credentials"])
    for m_id in measurement_ids:
        if download_preproc:
            logger.info("Downloading preprocessed files for '%s'" % m_id)
            scc.download_preprocessed(m_id)
            logger.info("Complete")
        if download_optical:
            logger.info("Downloading optical files for '%s'" % m_id)
            scc.download_optical(m_id)
            logger.info("Complete")
        if download_graph:
            logger.info("Downloading profile graph files for '%s'" % m_id)
            scc.download_graphs(m_id)
            logger.info("Complete")


def settings_from_path(config_file_path):
    """Read the configuration file.

    The file should be in YAML syntax."""

    if not os.path.isfile(config_file_path):
        raise argparse.ArgumentTypeError(
            "Wrong path for configuration file (%s)" % config_file_path
        )

    with open(config_file_path) as yaml_file:
        try:
            settings = yaml.safe_load(yaml_file)
            logger.debug("Read settings file(%s)" % config_file_path)
        except Exception:
            raise argparse.ArgumentTypeError(
                "Could not parse YAML file (%s)" % config_file_path
            )

    # YAML limitation: does not read tuples
    settings["basic_credentials"] = tuple(settings["basic_credentials"])
    settings["website_credentials"] = tuple(settings["website_credentials"])
    return settings


# Setup for command specific parsers
def setup_delete(parser):
    def delete_from_args(parsed):
        delete_measurements(parsed.IDs, parsed.config)

    parser.add_argument("IDs", nargs="+", help="measurement IDs to delete.")
    parser.set_defaults(execute=delete_from_args)


def setup_rerun_all(parser):
    def rerun_all_from_args(parsed):
        rerun_all(parsed.IDs, parsed.process, parsed.config)

    parser.add_argument("IDs", nargs="+", help="Measurement IDs to rerun.")
    parser.add_argument(
        "-p",
        "--process",
        help="Wait for the results of the processing.",
        action="store_true",
    )
    parser.set_defaults(execute=rerun_all_from_args)


def setup_rerun_processing(parser):
    def rerun_processing_from_args(parsed):
        rerun_processing(parsed.IDs, parsed.process, parsed.config)

    parser.add_argument(
        "IDs", nargs="+", help="Measurement IDs to rerun the processing on."
    )
    parser.add_argument(
        "-p",
        "--process",
        help="Wait for the results of the processing.",
        action="store_true",
    )
    parser.set_defaults(execute=rerun_processing_from_args)


def setup_process_file(parser):
    """Upload and monitor processing progress."""

    def process_file_from_args(parsed):
        process_file(
            parsed.filename,
            parsed.system,
            parsed.config,
            monitor=True,
            rs_filename=parsed.radiosounding,
            ov_filename=parsed.overlap,
            lr_filename=parsed.lidarratio,
        )

    parser.add_argument("filename", help="Measurement file name or path.")
    parser.add_argument("system", help="Processing system id.")
    parser.add_argument(
        "--radiosounding", default=None, help="Radiosounding file name or path"
    )
    parser.add_argument("--overlap", default=None, help="Overlap file name or path")
    parser.add_argument(
        "--lidarratio", default=None, help="Lidar ratio file name or path"
    )

    parser.set_defaults(execute=process_file_from_args)


def setup_upload_file(parser):
    """Upload but do not monitor processing progress."""

    def upload_file_from_args(parsed):
        process_file(
            parsed.filename,
            parsed.system,
            parsed.config,
            monitor=False,
            rs_filename=parsed.radiosounding,
            ov_filename=parsed.overlap,
            lr_filename=parsed.lidarratio,
        )

    parser.add_argument("filename", help="Measurement file name or path.")
    parser.add_argument("system", help="Processing system id.")
    parser.add_argument(
        "--radiosounding", default=None, help="Radiosounding file name or path"
    )
    parser.add_argument("--overlap", default=None, help="Overlap file name or path")
    parser.add_argument(
        "--lidarratio", default=None, help="Lidar ratio file name or path"
    )

    parser.set_defaults(execute=upload_file_from_args)


def setup_list_measurements(parser):
    def list_measurements_from_args(parsed):
        list_measurements(
            parsed.config,
            station=parsed.station,
            system=parsed.system,
            start=parsed.start,
            stop=parsed.stop,
            upload_status=parsed.upload_status,
            preprocessing_status=parsed.preprocessing_status,
            optical_processing=parsed.optical_processing_status,
        )

    def status(arg):
        if -127 <= int(arg) <= 127:
            return arg
        else:
            raise argparse.ArgumentTypeError("Status must be between -127 and 127")

    def date(arg):
        if re.match(r"\d{4}-\d{2}-\d{2}", arg):
            return arg
        else:
            raise argparse.ArgumentTypeError("Date must be in format 'YYYY-MM-DD'")

    parser.add_argument("--station", help="Filter for only the selected station")
    parser.add_argument("--system", help="Filter for only the selected station")
    parser.add_argument(
        "--start", help="Filter for only the selected station", type=date
    )
    parser.add_argument(
        "--stop", help="Filter for only the selected station", type=date
    )
    parser.add_argument(
        "--upload-status", help="Filter for only the selected station", type=status
    )
    parser.add_argument(
        "--preprocessing-status",
        help="Filter for only the selected station",
        type=status,
    )
    parser.add_argument(
        "--optical-processing-status",
        help="Filter for only the selected station",
        type=status,
    )
    parser.set_defaults(execute=list_measurements_from_args)


def setup_download_measurements(parser):
    def download_measurements_from_args(parsed):
        preproc = parsed.download_preprocessed
        optical = parsed.download_optical
        graphs = parsed.download_profile_graphs
        if not preproc and not graphs:
            optical = True
        download_measurements(parsed.IDs, preproc, optical, graphs, parsed.config)

    parser.add_argument(
        "IDs", help="Measurement IDs that should be downloaded.", nargs="+"
    )
    parser.add_argument(
        "--download-preprocessed",
        action="store_true",
        help="Download preprocessed files.",
    )
    parser.add_argument(
        "--download-optical",
        action="store_true",
        help="Download optical files (default if no other download is used).",
    )
    parser.add_argument(
        "--download-profile-graphs",
        action="store_true",
        help="Download profile graph files.",
    )
    parser.set_defaults(execute=download_measurements_from_args)


def main():
    # Define the command line arguments.
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    delete_parser = subparsers.add_parser("delete", help="Deletes a measurement.")
    rerun_all_parser = subparsers.add_parser(
        "rerun-all", help="Reprocess a measurement on the SCC."
    )
    rerun_processing_parser = subparsers.add_parser(
        "rerun-processing", help="Rerun processing routines for a measurement."
    )
    process_file_parser = subparsers.add_parser(
        "process-file", help="Upload a file and download processing results."
    )
    upload_file_parser = subparsers.add_parser("upload-file", help="Upload a file.")
    list_parser = subparsers.add_parser(
        "list", help="List measurements registered on the SCC."
    )
    download_parser = subparsers.add_parser(
        "download", help="Download selected measurements."
    )

    setup_delete(delete_parser)
    setup_rerun_all(rerun_all_parser)
    setup_rerun_processing(rerun_processing_parser)
    setup_process_file(process_file_parser)
    setup_upload_file(upload_file_parser)
    setup_list_measurements(list_parser)
    setup_download_measurements(download_parser)

    # Verbosity settings from http://stackoverflow.com/a/20663028
    parser.add_argument(
        "-d",
        "--debug",
        help="Print debugging information.",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    parser.add_argument(
        "-s",
        "--silent",
        help="Show only warning and error messages.",
        action="store_const",
        dest="loglevel",
        const=logging.WARNING,
    )

    # Setup default config location
    home = os.path.expanduser("~")
    default_config_location = os.path.abspath(os.path.join(home, ".scc_access.yaml"))
    parser.add_argument(
        "-c",
        "--config",
        help="Path to the config file.",
        type=settings_from_path,
        default=default_config_location,
    )

    args = parser.parse_args()

    # Get the logger with the appropriate level
    logging.basicConfig(format="%(levelname)s: %(message)s", level=args.loglevel)

    # Dispatch to appropriate function
    args.execute(args)


# When running through terminal
if __name__ == "__main__":
    main()
