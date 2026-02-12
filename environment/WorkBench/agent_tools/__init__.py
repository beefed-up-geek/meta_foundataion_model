"""WorkBench Tool Module - LangChain Tool Wrappers

This module provides LangChain-compatible tool specifications for LLM agents.
Each tool is a placeholder that should be connected to core.tools during execution.
"""

from .tools import (
    ALL_TOOLS,
    TOOL_MAPPING,
    # Calendar tools
    calendar_search_events,
    calendar_create_event,
    calendar_get_event_information_by_id,
    calendar_update_event,
    calendar_delete_event,
    # Email tools
    email_search_emails,
    email_send_email,
    email_get_email_information_by_id,
    email_forward_email,
    email_reply_email,
    email_delete_email,
    # Analytics tools
    analytics_create_plot,
    analytics_total_visits_count,
    analytics_engaged_users_count,
    analytics_traffic_source_count,
    analytics_get_average_session_duration,
    analytics_get_visitor_information_by_id,
    # Project Management tools
    project_management_search_tasks,
    project_management_create_task,
    project_management_get_task_information_by_id,
    project_management_update_task,
    project_management_delete_task,
    # CRM tools
    customer_relationship_manager_search_customers,
    customer_relationship_manager_add_customer,
    customer_relationship_manager_update_customer,
    customer_relationship_manager_delete_customer,
    # Company Directory tools
    company_directory_find_email_address,
)

__all__ = [
    "ALL_TOOLS",
    "TOOL_MAPPING",
    "calendar_search_events",
    "calendar_create_event",
    "calendar_get_event_information_by_id",
    "calendar_update_event",
    "calendar_delete_event",
    "email_search_emails",
    "email_send_email",
    "email_get_email_information_by_id",
    "email_forward_email",
    "email_reply_email",
    "email_delete_email",
    "analytics_create_plot",
    "analytics_total_visits_count",
    "analytics_engaged_users_count",
    "analytics_traffic_source_count",
    "analytics_get_average_session_duration",
    "analytics_get_visitor_information_by_id",
    "project_management_search_tasks",
    "project_management_create_task",
    "project_management_get_task_information_by_id",
    "project_management_update_task",
    "project_management_delete_task",
    "customer_relationship_manager_search_customers",
    "customer_relationship_manager_add_customer",
    "customer_relationship_manager_update_customer",
    "customer_relationship_manager_delete_customer",
    "company_directory_find_email_address",
]
