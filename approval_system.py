# approval_system.py - Email approval system with confirmation emails
import sqlite3
import json
import uuid
import logging
import os
import base64
import re
import requests
from datetime import datetime
from azure.identity import ClientSecretCredential
from config import *

log = logging.getLogger("approval-system")

class GraphEmailSender:
    def __init__(self):
        """Initialize Microsoft Graph client for sending emails"""
        try:
            credential = ClientSecretCredential(
                tenant_id=AZURE_TENANT_ID,
                client_id=AZURE_CLIENT_ID,
                client_secret=AZURE_CLIENT_SECRET
            )
            
            scopes = ['https://graph.microsoft.com/.default']
            self.access_token = credential.get_token(scopes[0]).token
            self.sender_email = MONITOR_EMAIL
            self.graph_url = "https://graph.microsoft.com/v1.0"
            log.info("✓ Microsoft Graph email sender initialized")
            
        except Exception as e:
            log.error(f"Failed to initialize Graph client: {e}")
            raise
    
    def send_email_with_attachment(self, to_email: str, subject: str, html_body: str, 
                                   attachment_path: str = None, attachment_filename: str = None):
        """Send email with optional attachment"""
        try:
            # Check if this is a system-generated email to prevent loops
            if "APPROVAL REQUIRED" in subject and "Invoice" in subject:
                log.info(f"System is sending approval email: {subject}")
            
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            message = {
                "message": {
                    "subject": subject,
                    "body": {
                        "contentType": "html",
                        "content": html_body
                    },
                    "toRecipients": [
                        {
                            "emailAddress": {
                                "address": to_email
                            }
                        }
                    ]
                },
                "saveToSentItems": "true"
            }
            
            # Add attachment if provided
            if attachment_path and os.path.exists(attachment_path):
                with open(attachment_path, 'rb') as file:
                    file_data = file.read()
                
                encoded_data = base64.b64encode(file_data).decode('utf-8')
                
                attachment = {
                    "@odata.type": "#microsoft.graph.fileAttachment",
                    "name": attachment_filename or os.path.basename(attachment_path),
                    "contentType": self._get_content_type(attachment_path),
                    "contentBytes": encoded_data
                }
                
                message["message"]["attachments"] = [attachment]
                log.info(f"Added attachment: {attachment['name']}")
            
            # Send the email
            url = f"{self.graph_url}/users/{self.sender_email}/sendMail"
            response = requests.post(url, headers=headers, json=message)
            
            if response.status_code in [200, 202]:
                log.info(f"✓ Email sent to {to_email}")
                return True
            else:
                log.error(f"Failed to send email: {response.status_code} - {response.text}")
                return False
            
        except Exception as e:
            log.error(f"Failed to send email: {e}")
            return False
    
    def _get_content_type(self, filepath: str) -> str:
        """Determine content type based on file extension"""
        ext = os.path.splitext(filepath)[1].lower()
        content_types = {
            '.pdf': 'application/pdf',
            '.doc': 'application/msword',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.xls': 'application/vnd.ms-excel',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.txt': 'text/plain',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png'
        }
        return content_types.get(ext, 'application/octet-stream')

# Global email sender instance
_email_sender = None

def get_email_sender():
    """Get or create the Graph email sender instance"""
    global _email_sender
    if _email_sender is None:
        _email_sender = GraphEmailSender()
    return _email_sender

def init_database():
    """Initialize SQLite database for approvals"""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS approvals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            approval_id TEXT UNIQUE,
            file_path TEXT,
            extracted_data TEXT,
            sap_payload TEXT,
            status TEXT DEFAULT 'pending',
            sap_reference TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processed_at TIMESTAMP,
            email_subject TEXT,
            approver_email TEXT,
            requester_email TEXT,
            is_system_generated BOOLEAN DEFAULT 0
        )
    ''')
    
    conn.commit()
    conn.close()
    log.info("Approval database initialized")

def get_approver_email(extracted_data):
    """Determine approver email based on amount"""
    try:
        total_payable = extracted_data.get("total_payable", "0")
        
        # Convert to float safely
        amount = 0
        if total_payable and total_payable != "N/A" and total_payable != "":
            try:
                amount_str = str(total_payable)
                amount_str = re.sub(r'[^\d.,-]', '', amount_str)
                amount_str = amount_str.replace(',', '')
                amount = float(amount_str) if amount_str and amount_str != "-" else 0
            except Exception as conv_err:
                log.error(f"Error converting amount: {conv_err}")
                amount = 0
        
        log.info(f"Amount: {amount}")
        
        # Simple logic: < 100 goes to small approver, >= 100 goes to large approver
        if 0 < amount < 101:
            log.info(f"Sending to SMALL amount approver (<100): {APPROVER_SMALL_AMOUNT}")
            return APPROVER_SMALL_AMOUNT
        else:
            log.info(f"Sending to LARGE amount approver (>100): {APPROVER_LARGE_AMOUNT}")
            return APPROVER_LARGE_AMOUNT
            
    except Exception as e:
        log.error(f"Error determining approver: {e}")
        return APPROVER_LARGE_AMOUNT

def save_for_approval(file_path, extracted_data, sap_payload, email_subject=""):
    """Save invoice for approval with dynamic approver"""
    approval_id = str(uuid.uuid4())[:8]
    
    # Determine approver based on amount
    approver_email = get_approver_email(extracted_data)
    
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO approvals (approval_id, file_path, extracted_data, sap_payload, email_subject, approver_email, requester_email)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        approval_id,
        file_path,
        json.dumps(extracted_data),
        json.dumps(sap_payload),
        email_subject,
        approver_email,
        MONITOR_EMAIL  # Store requester email
    ))
    
    conn.commit()
    conn.close()
    
    log.info(f"Saved for approval: {approval_id} -> Approver: {approver_email}")
    return approval_id

def send_approval_email(approval_id, extracted_data):
    """Send approval email with buttons and invoice attachment"""
    try:
        # Get the approver email and attachment path from database
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute('SELECT file_path, approver_email FROM approvals WHERE approval_id = ?', (approval_id,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            log.error(f"No record found for approval {approval_id}")
            return False
            
        attachment_path, approver_email = result
        
        if not approver_email:
            log.error(f"No approver email found for {approval_id}")
            return False
        
        # Create approval URLs
        approve_url = f"{SERVER_URL}/approve/{approval_id}"
        deny_url = f"{SERVER_URL}/deny/{approval_id}"
        
        # Get invoice details
        total_amount = extracted_data.get('total_payable', 'N/A')
        currency = extracted_data.get('currency', '')
        invoice_number = extracted_data.get('invoice_number', 'Unknown')
        
        if invoice_number == 'Unknown' or invoice_number == '':
            log.error(f"❌ Cannot send approval email - no invoice number extracted!")
            return False
        
        # HTML email content
        html_content = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; color: white; border-radius: 10px 10px 0 0;">
                <h1 style="margin: 0;">Invoice Approval Required</h1>
            </div>
            
            <div style="padding: 20px; background-color: #f9f9f9;">
                <div style="background-color: white; padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #667eea;">
                    <h3 style="color: #333; margin-top: 0;">Invoice Details</h3>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr>
                            <td style="padding: 8px 0; border-bottom: 1px solid #eee; width: 40%;"><strong>Invoice Number:</strong></td>
                            <td style="padding: 8px 0; border-bottom: 1px solid #eee;">{invoice_number}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; border-bottom: 1px solid #eee;"><strong>Invoice Date:</strong></td>
                            <td style="padding: 8px 0; border-bottom: 1px solid #eee;">{extracted_data.get('invoice_date', 'N/A')}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; border-bottom: 1px solid #eee;"><strong>Supplier:</strong></td>
                            <td style="padding: 8px 0; border-bottom: 1px solid #eee;">{extracted_data.get('supplier', {}).get('name', 'N/A')}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; border-bottom: 1px solid #eee;"><strong>Total Amount:</strong></td>
                            <td style="padding: 8px 0; border-bottom: 1px solid #eee; font-weight: bold; color: #2c3e50;">{total_amount if total_amount != 'N/A' else 'Not extracted'} {currency}</td>
                        </tr>
                    </table>
                </div>

                <p style="color: #555; margin: 25px 0;">
                    <strong>Invoice attachment is included with this email.</strong>
                </p>

                <div style="text-align: center; margin: 30px 0;">
                    <a href="{approve_url}" style="
                        background-color: #4CAF50;
                        color: white;
                        padding: 14px 28px;
                        text-decoration: none;
                        border-radius: 6px;
                        font-weight: bold;
                        margin-right: 15px;
                        font-size: 16px;
                        display: inline-block;
                        box-shadow: 0 4px 6px rgba(76, 175, 80, 0.3);
                    ">APPROVE</a>
                    
                    <a href="{deny_url}" style="
                        background-color: #f44336;
                        color: white;
                        padding: 14px 28px;
                        text-decoration: none;
                        border-radius: 6px;
                        font-weight: bold;
                        font-size: 16px;
                        display: inline-block;
                        box-shadow: 0 4px 6px rgba(244, 67, 54, 0.3);
                    ">DENY</a>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Get email sender and send email
        email_sender = get_email_sender()
        
        # Build subject
        subject = f"APPROVAL REQUIRED: Invoice {invoice_number}"
        if total_amount != 'N/A' and total_amount != '' and currency:
            subject += f" ({total_amount} {currency})"
        
        success = email_sender.send_email_with_attachment(
            to_email=approver_email,
            subject=subject,
            html_body=html_content,
            attachment_path=attachment_path if os.path.exists(attachment_path) else None,
            attachment_filename=os.path.basename(attachment_path) if attachment_path else None
        )
        
        if success:
            log.info(f"✅ Approval email sent to {approver_email} for {approval_id}")
        else:
            log.error(f"❌ Failed to send approval email for {approval_id}")
            
        return success
        
    except Exception as e:
        log.error(f"❌ Failed to send approval email: {e}")
        return False

def send_decision_email(approval_id, status, sap_reference=None):
    """Send confirmation email to requester about approval decision - FIXED"""
    try:
        # Get approval details from database
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT extracted_data, requester_email, approver_email, sap_reference
            FROM approvals WHERE approval_id = ?
        ''', (approval_id,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            log.error(f"No record found for approval {approval_id}")
            return False
            
        extracted_data_json, requester_email, approver_email, db_sap_reference = result
        
        # Use provided sap_reference or the one from database
        if not sap_reference:
            sap_reference = db_sap_reference
        
        if not requester_email:
            requester_email = MONITOR_EMAIL
        
        # Parse extracted data
        extracted_data = json.loads(extracted_data_json) if extracted_data_json else {}
        
        # Get invoice details
        invoice_number = extracted_data.get('invoice_number', 'Unknown')
        invoice_date = extracted_data.get('invoice_date', 'N/A')
        supplier = extracted_data.get('supplier', {}).get('name', 'N/A')
        total_amount = extracted_data.get('total_payable', 'N/A')
        currency = extracted_data.get('currency', '')
        
        # Status color and text
        if status == 'approved':
            status_color = "#4CAF50"
            status_text = "APPROVED"
            action = "approved"
        else:
            status_color = "#f44336"
            status_text = "REJECTED"
            action = "rejected"
        
        # **FIXED: Show SAP reference properly**
        sap_ref_display = sap_reference if sap_reference else 'Not yet generated'
        
        # HTML email content - UPDATED
        html_content = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background: {status_color}; padding: 20px; color: white; border-radius: 10px 10px 0 0;">
                <h1 style="margin: 0;">Invoice {status_text}</h1>
            </div>
            
            <div style="padding: 20px; background-color: #f9f9f9;">
                <div style="background-color: white; padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid {status_color};">
                    <h3 style="color: #333; margin-top: 0;">Invoice Decision Details</h3>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr>
                            <td style="padding: 8px 0; border-bottom: 1px solid #eee; width: 40%;"><strong>Invoice Number:</strong></td>
                            <td style="padding: 8px 0; border-bottom: 1px solid #eee;">{invoice_number}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; border-bottom: 1px solid #eee;"><strong>Status:</strong></td>
                            <td style="padding: 8px 0; border-bottom: 1px solid #eee; font-weight: bold; color: {status_color};">{status_text}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; border-bottom: 1px solid #eee;"><strong>SAP Reference:</strong></td>
                            <td style="padding: 8px 0; border-bottom: 1px solid #eee; font-weight: bold; color: #2c3e50;">{sap_ref_display}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; border-bottom: 1px solid #eee;"><strong>Invoice Date:</strong></td>
                            <td style="padding: 8px 0; border-bottom: 1px solid #eee;">{invoice_date}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; border-bottom: 1px solid #eee;"><strong>Supplier:</strong></td>
                            <td style="padding: 8px 0; border-bottom: 1px solid #eee;">{supplier}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; border-bottom: 1px solid #eee;"><strong>Total Amount:</strong></td>
                            <td style="padding: 8px 0; border-bottom: 1px solid #eee; font-weight: bold;">{total_amount if total_amount != 'N/A' else 'Not extracted'} {currency}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; border-bottom: 1px solid #eee;"><strong>Approved/Rejected By:</strong></td>
                            <td style="padding: 8px 0; border-bottom: 1px solid #eee;">{approver_email}</td>
                        </tr>
                    </table>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Get email sender and send email
        email_sender = get_email_sender()
        
        subject = f"Invoice {invoice_number} {status_text}"
        if sap_reference:
            subject += f" (SAP Ref: {sap_reference})"
        
        success = email_sender.send_email_with_attachment(
            to_email=requester_email,
            subject=subject,
            html_body=html_content
        )
        
        if success:
            log.info(f"✅ Decision email sent to {requester_email} for {approval_id}")
            if sap_reference:
                log.info(f"📋 SAP Reference included: {sap_reference}")
        else:
            log.error(f"❌ Failed to send decision email for {approval_id}")
            
        return success
        
    except Exception as e:
        log.error(f"❌ Failed to send decision email: {e}")
        return False

def get_approval_status(approval_id):
    """Get approval status"""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    cursor.execute('SELECT status, sap_payload, sap_reference FROM approvals WHERE approval_id = ?', (approval_id,))
    result = cursor.fetchone()
    
    conn.close()
    
    if result:
        return {
            "status": result[0], 
            "sap_payload": json.loads(result[1]) if result[1] else None,
            "sap_reference": result[2]
        }
    return None

def update_approval_status(approval_id, status, sap_reference=None):
    """Update approval status and send confirmation email - IMPROVED"""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    # If we have a SAP reference, update it
    if sap_reference:
        cursor.execute('''
            UPDATE approvals 
            SET status = ?, sap_reference = ?, processed_at = CURRENT_TIMESTAMP
            WHERE approval_id = ?
        ''', (status, sap_reference, approval_id))
        log.info(f"Updated approval {approval_id} to {status} with SAP ref: {sap_reference}")
    else:
        cursor.execute('''
            UPDATE approvals 
            SET status = ?, processed_at = CURRENT_TIMESTAMP
            WHERE approval_id = ?
        ''', (status, approval_id))
        log.info(f"Updated approval {approval_id} to {status} (no SAP ref)")
    
    conn.commit()
    conn.close()
    
    # Send confirmation email to requester - WITH SAP REFERENCE
    send_decision_email(approval_id, status, sap_reference)
    
    return True

def get_sap_reference(approval_id):
    """Get SAP reference from database if exists"""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('SELECT sap_reference FROM approvals WHERE approval_id = ?', (approval_id,))
    result = cursor.fetchone()
    conn.close()
    
    return result[0] if result else None