import os
import time
import threading
import logging
import requests
import json
import asyncio
from datetime import datetime, timedelta
from azure.identity import ClientSecretCredential
from msgraph import GraphServiceClient
from msgraph.generated.models.message import Message
from msgraph.generated.users.item.messages.messages_request_builder import MessagesRequestBuilder
import base64

log = logging.getLogger("email-monitor")

class GraphAPIMonitor:
    def __init__(self, check_interval=60):
        self.check_interval = check_interval
        self.running = False
        self.processed_emails = set()
        self.graph_client = None
        self.loop = None 
        self.server_url = "http://localhost:3002"

        from config import (
            AZURE_TENANT_ID,
            AZURE_CLIENT_ID,
            AZURE_CLIENT_SECRET,
            MONITOR_EMAIL
        )

        self.tenant_id = AZURE_TENANT_ID
        self.client_id = AZURE_CLIENT_ID
        self.client_secret = AZURE_CLIENT_SECRET
        self.user_email = MONITOR_EMAIL
    
    def _get_graph_client(self):
        """Authenticate and get Microsoft Graph client"""
        try:
            credential = ClientSecretCredential(
                tenant_id=self.tenant_id,
                client_id=self.client_id,
                client_secret=self.client_secret
            )
            
            # Create Graph client with the required permissions
            scopes = ['https://graph.microsoft.com/.default']
            self.graph_client = GraphServiceClient(credential, scopes)
            log.info("Successfully authenticated with Microsoft Graph API")
            return True
            
        except Exception as e:
            log.error(f"Failed to authenticate with Graph API: {e}")
            return False
    
    def start(self):
        """Start monitoring emails via Graph API"""
        if not self._get_graph_client():
            return False
            
        self.running = True
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._run_async_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        log.info("✓ Microsoft Graph API monitoring started")
        log.info(f"✓ Monitoring email: {self.user_email}")
        log.info(f"✓ Checking every {self.check_interval} seconds")
        return True
    
    def _run_async_monitor(self):
        """Run async monitor in a separate thread with its own event loop"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        # Run the async monitor loop
        self.loop.run_until_complete(self._async_monitor_loop())
    
    async def _async_monitor_loop(self):
        """Async monitoring loop"""
        while self.running:
            try:
                await self._check_new_emails()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                log.error(f"Monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _check_new_emails(self):
        """Check for new emails with invoice attachments"""
        try:
            # Calculate time threshold (e.g., last 5 minutes)
            time_threshold = datetime.utcnow() - timedelta(minutes=5)
            
            # Query Graph API for new emails
            query_params = MessagesRequestBuilder.MessagesRequestBuilderGetQueryParameters(
                filter=f"receivedDateTime ge {time_threshold.isoformat()}Z and hasAttachments eq true",
                select=["id", "subject", "sender", "receivedDateTime", "hasAttachments"],
                top=20,
                orderby="receivedDateTime DESC"
            )
            
            request_config = MessagesRequestBuilder.MessagesRequestBuilderGetRequestConfiguration(
                query_parameters=query_params,
            )
            
            # Get messages (await the async call)
            messages_response = await self.graph_client.users.by_user_id(self.user_email).messages.get(
                request_configuration=request_config
            )
            
            if messages_response and messages_response.value:
                for message in messages_response.value:
                    await self._process_email_message(message)
                        
        except Exception as e:
            log.error(f"Error checking emails: {e}")

    async def _mark_email_unread(self, email_id: str):
        """Mark email as UNREAD in Outlook - SIMPLIFIED VERSION"""
        try:
            # Use simple dictionary instead of UpdateMessage class
            update_data = {
                "isRead": True
            }
            
            # Update the message
            await (
                self.graph_client
                .users
                .by_user_id(self.user_email)
                .messages
                .by_message_id(email_id)
                .patch(update_data)
            )

            log.info(f"Marked email as UNREAD: {email_id}")

        except Exception as e:
            log.error(f"Failed to mark email unread: {e}")
    
    async def _process_email_message(self, message):
        """Process a single email message - WITH INFINITE LOOP PREVENTION"""
        try:
            email_id = message.id
            
            # Skip already processed emails
            if email_id in self.processed_emails:
                return
                
            subject = (message.subject or "").lower()
            
            # Get sender information
            sender_address = ""
            if message.sender and message.sender.email_address:
                sender_address = message.sender.email_address.address.lower()
            
            # CRITICAL FIX: Skip emails that are from our own system or are approval emails
            # This prevents infinite loops
            
            # Check 1: Skip emails from our own system (self)
            if sender_address == self.user_email.lower():
                log.info(f"⚠️ Skipping email from self (system): {message.subject[:50]}...")
                self.processed_emails.add(email_id)
                return
            
            # Check 2: Skip emails that are approval requests (subject contains "APPROVAL REQUIRED")
            if "approval required" in subject:
                log.info(f"⚠️ Skipping approval email to prevent loop: {message.subject[:50]}...")
                self.processed_emails.add(email_id)
                return
            
            # Check 3: Skip emails from the system itself (Exchange system address)
            if sender_address and ("exchangelabs" in sender_address.lower() or 
                                 "exchange administrative group" in sender_address.lower()):
                log.info(f"⚠️ Skipping system-generated email: {message.subject[:50]}...")
                self.processed_emails.add(email_id)
                return
            
            # Check 4: Only process if subject contains "invoice" AND not from our system
            # Also check that it's a real invoice, not a system notification
            if "invoice" in subject:
                log.info(f"📧 Found external invoice email: {message.subject}")
                log.info(f"   From: {sender_address}")
                log.info(f"   Date: {message.received_date_time}")
                
                # Fetch attachments explicitly
                attachments_response = await (
                    self.graph_client
                    .users
                    .by_user_id(self.user_email)
                    .messages
                    .by_message_id(email_id)
                    .attachments
                    .get()
                )

                if attachments_response and attachments_response.value:
                    await self._process_attachments(attachments_response.value, subject)
                else:
                    log.warning(f"No attachments found for invoice email: {message.subject}")
                
                # Mark email as unread (optional - you can remove this if not needed)
                # await self._mark_email_unread(email_id)

                # Mark as processed
                self.processed_emails.add(email_id)
                
        except Exception as e:
            log.error(f"Error processing email {message.id}: {e}")
    
    async def _process_attachments(self, attachments, email_subject):
        """Process attachments from email"""
        attachments_folder = "email_attachments"
        os.makedirs(attachments_folder, exist_ok=True)

        invoice_found = False
        
        for attachment in attachments:
            if attachment.odata_type == "#microsoft.graph.fileAttachment":
                filename = attachment.name.lower()

                # Only process invoice-related files
                if filename.endswith(('.pdf', '.doc', '.docx', '.xls', '.xlsx')):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_filename = f"{timestamp}_{attachment.name}"
                    save_path = os.path.join(attachments_folder, save_filename)

                    if attachment.content_bytes:
                        # Base64 decode
                        try:
                            file_bytes = base64.b64decode(attachment.content_bytes)
                            
                            with open(save_path, "wb") as f:
                                f.write(file_bytes)

                            log.info(f"💾 Saved attachment: {save_path}")
                            await self._trigger_processing(save_path, email_subject)
                            invoice_found = True
                            
                        except Exception as e:
                            log.error(f"Failed to save attachment {filename}: {e}")
        
        if not invoice_found:
            log.warning("No invoice attachments found in email")
    
    async def _trigger_processing(self, file_path, subject):
        """Trigger processing via HTTP"""
        try:
            # Use a thread pool executor for the synchronous requests call
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._sync_trigger_processing, file_path, subject)
        except Exception as e:
            log.error(f"Error in trigger_processing: {e}")
    
    # def _sync_trigger_processing(self, file_path, subject):
    #     """Synchronous version of trigger_processing"""
    #     try:
    #         with open(file_path, 'rb') as f:
    #             files = {'file': (os.path.basename(file_path), f)}
    #             response = requests.post(
    #                 "http://localhost:3002/process-email-file",
    #                 files=files,
    #                 data={'subject': subject},
    #                 timeout=30
    #             )
    #             if response.status_code == 200:
    #                 log.info(f"✅ Processing triggered for: {os.path.basename(file_path)}")
    #             else:
    #                 log.error(f"❌ Failed to trigger processing: {response.text}")
    #     except Exception as e:
    #         log.error(f"Error in sync_trigger_processing: {e}")

    # In email_monitor.py, change the _sync_trigger_processing function:
    def _sync_trigger_processing(self, file_path, subject):
        """Synchronous version of trigger_processing - FIXED"""
        try:
            # Wait for server to be ready
            time.sleep(1)
            
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f, 'application/pdf')}
                data = {'subject': subject}
                
                log.info(f"📤 Sending to {self.server_url}/process-email-file")
                
                # ✅ Call /process-email-file (not /extract)
                response = requests.post(
                    f"{self.server_url}/process-email-file",  # NOT /extract
                    files=files,
                    data=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    log.info(f"✅ Processing triggered for: {os.path.basename(file_path)}")
                else:
                    log.error(f"❌ Failed to trigger processing: {response.status_code} - {response.text}")
        except Exception as e:
            log.error(f"Error in sync_trigger_processing: {e}")
        
    def process_approval_replies():
        """Monitor for approval reply emails"""
        try:
            # Get emails that are replies to approval requests
            # Look for emails with "APPROVE" or "REJECT" in subject/body
            query = {
                "$filter": f"subject/body contains 'APPROVE' or subject/body contains 'REJECT'",
                "$top": 10
            }
            
            # Use Microsoft Graph API to check for replies
            # This needs to be integrated with your existing email monitoring
            
        except Exception as e:
            log.error(f"Error processing approval replies: {e}")
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        log.info("Microsoft Graph monitoring stopped")

# Global monitor instance
monitor = None

def start_email_monitoring():
    """Start the email monitoring system"""
    global monitor
    monitor = GraphAPIMonitor()
    return monitor.start()

def stop_email_monitoring():
    """Stop the email monitoring system"""
    global monitor
    if monitor:
        monitor.stop()


