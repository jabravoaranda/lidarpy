import pytz
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate
import smtplib
from email import utils

from loguru import logger

from lidarpy.utils.utils import LIDAR_INFO


def send_email(email_sender: dict, email_receiver: list[str], subject: str, email_content: str):
    """Send email to the specified email addresses

    Args:
        email_sender (dict): Dictionary with following keys: server, port, email_name, password.
        email_receiver (list[str]): List of email addresses to send the email to.
        email_content (str): Content of the email to be sent.

    Returns:
        _type_: _description_
    """
    
    # Check Input
    if isinstance(email_receiver, str):
        email_receiver = [email_receiver]

    logger.info("Send Email")    
    
    return_code = 1
    try:
        # Prepare Message
        message = MIMEMultipart()
        message["To"] = ", ".join(email_receiver)
        message["To"] = ", ".join(email_receiver)
        message["Date"] = formatdate(localtime=True)
        message["Subject"] = subject
        message.attach(MIMEText(email_content))

        # Connect to email server and send email
        # ssl_context = ssl.create_default_context()
        # conn = smtplib.SMTP_SSL(email_sender['server'], email_sender['port'], context=ssl_context)
        conn = smtplib.SMTP(email_sender["server"], email_sender["port"])
        conn.ehlo()
        conn.starttls()
        conn.ehlo()
        conn.login(email_sender["email_name"], email_sender["password"])
        conn.sendmail(email_sender["email_name"], email_receiver, message.as_string())
        logger.info("Email Sent")
    except:
        logger.error("Email not sent")
        return_code = 0
    finally:
        conn.quit()
    return return_code