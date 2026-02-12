"""Core tool implementations with workspace management

Each tool:
- Input: workspace_id + original tool parameters
- Output: (new_workspace_id, success, result)
- Internally calls workspace_manager.fork_and_execute()
"""

from typing import Tuple, Any, Optional
from .workspace_manager import WorkspaceManager
from .tool_executor import execute_action


# Singleton workspace manager instance
_workspace_manager: Optional[WorkspaceManager] = None


def initialize_tools(workspace_manager: WorkspaceManager):
    """Initialize tools with a workspace manager instance"""
    global _workspace_manager
    _workspace_manager = workspace_manager


def _execute_tool(workspace_id: str, action: str) -> Tuple[str, bool, Any]:
    """Execute a tool action in a workspace

    Returns:
        (new_workspace_id, success, result)
    """
    if _workspace_manager is None:
        raise RuntimeError("Tools not initialized. Call initialize_tools() first.")

    new_id, success, result = _workspace_manager.fork_and_execute(
        workspace_id, action, execute_action
    )
    return new_id, success, result


# ============================================================================
# CALENDAR TOOLS (6)
# ============================================================================

def calendar_get_event_information_by_id(
    workspace_id: str,
    event_id: str = None,
    field: str = None
) -> Tuple[str, bool, Any]:
    """Returns the event for a given ID.

    Parameters:
        workspace_id: Current workspace ID
        event_id: 8-digit ID of the event
        field: Field to return (event_id, event_name, participant_email, event_start, duration)

    Returns:
        (new_workspace_id, success, result)
    """
    action = f'calendar.get_event_information_by_id.func(event_id="{event_id}", field="{field}")'
    return _execute_tool(workspace_id, action)


def calendar_search_events(
    workspace_id: str,
    query: str = "",
    time_min: str = None,
    time_max: str = None
) -> Tuple[str, bool, Any]:
    """Returns the events for a given query.

    Parameters:
        workspace_id: Current workspace ID
        query: Query to search for
        time_min: Lower bound for event's end time (YYYY-MM-DD HH:MM:SS)
        time_max: Upper bound for event's start time (YYYY-MM-DD HH:MM:SS)

    Returns:
        (new_workspace_id, success, result)
    """
    params = [f'query="{query}"']
    if time_min:
        params.append(f'time_min="{time_min}"')
    if time_max:
        params.append(f'time_max="{time_max}"')

    action = f'calendar.search_events.func({", ".join(params)})'
    return _execute_tool(workspace_id, action)


def calendar_create_event(
    workspace_id: str,
    event_name: str = None,
    participant_email: str = None,
    event_start: str = None,
    duration: str = None
) -> Tuple[str, bool, Any]:
    """Creates a new event.

    Parameters:
        workspace_id: Current workspace ID
        event_name: Name of the event
        participant_email: Email of the participant
        event_start: Start time (YYYY-MM-DD HH:MM:SS)
        duration: Duration in minutes

    Returns:
        (new_workspace_id, success, result)
    """
    action = f'calendar.create_event.func(event_name="{event_name}", participant_email="{participant_email}", event_start="{event_start}", duration="{duration}")'
    return _execute_tool(workspace_id, action)


def calendar_delete_event(
    workspace_id: str,
    event_id: str = None
) -> Tuple[str, bool, Any]:
    """Deletes an event.

    Parameters:
        workspace_id: Current workspace ID
        event_id: 8-digit ID of the event

    Returns:
        (new_workspace_id, success, result)
    """
    action = f'calendar.delete_event.func(event_id="{event_id}")'
    return _execute_tool(workspace_id, action)


def calendar_update_event(
    workspace_id: str,
    event_id: str = None,
    field: str = None,
    new_value: str = None
) -> Tuple[str, bool, Any]:
    """Updates an event.

    Parameters:
        workspace_id: Current workspace ID
        event_id: 8-digit ID of the event
        field: Field to update
        new_value: New value for the field

    Returns:
        (new_workspace_id, success, result)
    """
    action = f'calendar.update_event.func(event_id="{event_id}", field="{field}", new_value="{new_value}")'
    return _execute_tool(workspace_id, action)


# ============================================================================
# EMAIL TOOLS (6)
# ============================================================================

def email_get_email_information_by_id(
    workspace_id: str,
    email_id: str = None,
    field: str = None
) -> Tuple[str, bool, Any]:
    """Returns the email for a given ID.

    Parameters:
        workspace_id: Current workspace ID
        email_id: ID of the email
        field: Field to return

    Returns:
        (new_workspace_id, success, result)
    """
    action = f'email.get_email_information_by_id.func(email_id="{email_id}", field="{field}")'
    return _execute_tool(workspace_id, action)


def email_search_emails(
    workspace_id: str,
    query: str = "",
    date_min: str = None,
    date_max: str = None
) -> Tuple[str, bool, Any]:
    """Returns the emails for a given query.

    Parameters:
        workspace_id: Current workspace ID
        query: Query to search for
        date_min: Minimum date (YYYY-MM-DD)
        date_max: Maximum date (YYYY-MM-DD)

    Returns:
        (new_workspace_id, success, result)
    """
    params = [f'query="{query}"']
    if date_min:
        params.append(f'date_min="{date_min}"')
    if date_max:
        params.append(f'date_max="{date_max}"')

    action = f'email.search_emails.func({", ".join(params)})'
    return _execute_tool(workspace_id, action)


def email_send_email(
    workspace_id: str,
    recipient: str = None,
    subject: str = None,
    body: str = None
) -> Tuple[str, bool, Any]:
    """Sends an email.

    Parameters:
        workspace_id: Current workspace ID
        recipient: Email address of the recipient
        subject: Subject of the email
        body: Body of the email

    Returns:
        (new_workspace_id, success, result)
    """
    # Escape quotes in body
    if body:
        body = body.replace('"', '\\"')
    action = f'email.send_email.func(recipient="{recipient}", subject="{subject}", body="{body}")'
    return _execute_tool(workspace_id, action)


def email_delete_email(
    workspace_id: str,
    email_id: str = None
) -> Tuple[str, bool, Any]:
    """Deletes an email.

    Parameters:
        workspace_id: Current workspace ID
        email_id: ID of the email

    Returns:
        (new_workspace_id, success, result)
    """
    action = f'email.delete_email.func(email_id="{email_id}")'
    return _execute_tool(workspace_id, action)


def email_forward_email(
    workspace_id: str,
    email_id: str = None,
    recipient: str = None
) -> Tuple[str, bool, Any]:
    """Forwards an email.

    Parameters:
        workspace_id: Current workspace ID
        email_id: ID of the email
        recipient: Email address of the recipient

    Returns:
        (new_workspace_id, success, result)
    """
    action = f'email.forward_email.func(email_id="{email_id}", recipient="{recipient}")'
    return _execute_tool(workspace_id, action)


def email_reply_email(
    workspace_id: str,
    email_id: str = None,
    body: str = None
) -> Tuple[str, bool, Any]:
    """Replies to an email.

    Parameters:
        workspace_id: Current workspace ID
        email_id: ID of the email
        body: Body of the reply

    Returns:
        (new_workspace_id, success, result)
    """
    if body:
        body = body.replace('"', '\\"')
    action = f'email.reply_email.func(email_id="{email_id}", body="{body}")'
    return _execute_tool(workspace_id, action)


# ============================================================================
# ANALYTICS TOOLS (7)
# ============================================================================

def analytics_get_visitor_information_by_id(
    workspace_id: str,
    visitor_id: str = None
) -> Tuple[str, bool, Any]:
    """Returns visitor information for a given ID.

    Parameters:
        workspace_id: Current workspace ID
        visitor_id: ID of the visitor

    Returns:
        (new_workspace_id, success, result)
    """
    action = f'analytics.get_visitor_information_by_id.func(visitor_id="{visitor_id}")'
    return _execute_tool(workspace_id, action)


def analytics_create_plot(
    workspace_id: str,
    time_min: str = None,
    time_max: str = None,
    value_to_plot: str = None,
    plot_type: str = None
) -> Tuple[str, bool, Any]:
    """Creates a plot.

    Parameters:
        workspace_id: Current workspace ID
        time_min: Minimum time (YYYY-MM-DD)
        time_max: Maximum time (YYYY-MM-DD)
        value_to_plot: Value to plot
        plot_type: Type of plot (bar, line, etc.)

    Returns:
        (new_workspace_id, success, result)
    """
    action = f'analytics.create_plot.func(time_min="{time_min}", time_max="{time_max}", value_to_plot="{value_to_plot}", plot_type="{plot_type}")'
    return _execute_tool(workspace_id, action)


def analytics_total_visits_count(
    workspace_id: str,
    time_min: str = None,
    time_max: str = None
) -> Tuple[str, bool, Any]:
    """Returns total visits count.

    Parameters:
        workspace_id: Current workspace ID
        time_min: Minimum time (YYYY-MM-DD)
        time_max: Maximum time (YYYY-MM-DD)

    Returns:
        (new_workspace_id, success, result)
    """
    params = []
    if time_min:
        params.append(f'time_min="{time_min}"')
    if time_max:
        params.append(f'time_max="{time_max}"')

    action = f'analytics.total_visits_count.func({", ".join(params)})'
    return _execute_tool(workspace_id, action)


def analytics_engaged_users_count(
    workspace_id: str,
    time_min: str = None,
    time_max: str = None
) -> Tuple[str, bool, Any]:
    """Returns engaged users count.

    Parameters:
        workspace_id: Current workspace ID
        time_min: Minimum time (YYYY-MM-DD)
        time_max: Maximum time (YYYY-MM-DD)

    Returns:
        (new_workspace_id, success, result)
    """
    params = []
    if time_min:
        params.append(f'time_min="{time_min}"')
    if time_max:
        params.append(f'time_max="{time_max}"')

    action = f'analytics.engaged_users_count.func({", ".join(params)})'
    return _execute_tool(workspace_id, action)


def analytics_traffic_source_count(
    workspace_id: str,
    time_min: str = None,
    time_max: str = None,
    traffic_source: str = None
) -> Tuple[str, bool, Any]:
    """Returns traffic source count.

    Parameters:
        workspace_id: Current workspace ID
        time_min: Minimum time (YYYY-MM-DD)
        time_max: Maximum time (YYYY-MM-DD)
        traffic_source: Traffic source

    Returns:
        (new_workspace_id, success, result)
    """
    params = []
    if time_min:
        params.append(f'time_min="{time_min}"')
    if time_max:
        params.append(f'time_max="{time_max}"')
    if traffic_source:
        params.append(f'traffic_source="{traffic_source}"')

    action = f'analytics.traffic_source_count.func({", ".join(params)})'
    return _execute_tool(workspace_id, action)


def analytics_get_average_session_duration(
    workspace_id: str,
    time_min: str = None,
    time_max: str = None
) -> Tuple[str, bool, Any]:
    """Returns average session duration.

    Parameters:
        workspace_id: Current workspace ID
        time_min: Minimum time (YYYY-MM-DD)
        time_max: Maximum time (YYYY-MM-DD)

    Returns:
        (new_workspace_id, success, result)
    """
    params = []
    if time_min:
        params.append(f'time_min="{time_min}"')
    if time_max:
        params.append(f'time_max="{time_max}"')

    action = f'analytics.get_average_session_duration.func({", ".join(params)})'
    return _execute_tool(workspace_id, action)


# ============================================================================
# PROJECT MANAGEMENT TOOLS (6)
# ============================================================================

def project_management_get_task_information_by_id(
    workspace_id: str,
    task_id: str = None,
    field: str = None
) -> Tuple[str, bool, Any]:
    """Returns task information for a given ID.

    Parameters:
        workspace_id: Current workspace ID
        task_id: ID of the task
        field: Field to return

    Returns:
        (new_workspace_id, success, result)
    """
    action = f'project_management.get_task_information_by_id.func(task_id="{task_id}", field="{field}")'
    return _execute_tool(workspace_id, action)


def project_management_search_tasks(
    workspace_id: str,
    task_name: str = None,
    assigned_to_email: str = None,
    list_name: str = None,
    due_date: str = None,
    board: str = None
) -> Tuple[str, bool, Any]:
    """Searches for tasks.

    Parameters:
        workspace_id: Current workspace ID
        task_name: Name of the task
        assigned_to_email: Email of assignee
        list_name: Name of the list
        due_date: Due date (YYYY-MM-DD)
        board: Name of the board

    Returns:
        (new_workspace_id, success, result)
    """
    params = []
    if task_name:
        params.append(f'task_name="{task_name}"')
    if assigned_to_email:
        params.append(f'assigned_to_email="{assigned_to_email}"')
    if list_name:
        params.append(f'list_name="{list_name}"')
    if due_date:
        params.append(f'due_date="{due_date}"')
    if board:
        params.append(f'board="{board}"')

    action = f'project_management.search_tasks.func({", ".join(params)})'
    return _execute_tool(workspace_id, action)


def project_management_create_task(
    workspace_id: str,
    task_name: str = None,
    assigned_to_email: str = None,
    list_name: str = None,
    due_date: str = None,
    board: str = None
) -> Tuple[str, bool, Any]:
    """Creates a new task.

    Parameters:
        workspace_id: Current workspace ID
        task_name: Name of the task
        assigned_to_email: Email of assignee
        list_name: Name of the list
        due_date: Due date (YYYY-MM-DD)
        board: Name of the board

    Returns:
        (new_workspace_id, success, result)
    """
    action = f'project_management.create_task.func(task_name="{task_name}", assigned_to_email="{assigned_to_email}", list_name="{list_name}", due_date="{due_date}", board="{board}")'
    return _execute_tool(workspace_id, action)


def project_management_delete_task(
    workspace_id: str,
    task_id: str = None
) -> Tuple[str, bool, Any]:
    """Deletes a task.

    Parameters:
        workspace_id: Current workspace ID
        task_id: ID of the task

    Returns:
        (new_workspace_id, success, result)
    """
    action = f'project_management.delete_task.func(task_id="{task_id}")'
    return _execute_tool(workspace_id, action)


def project_management_update_task(
    workspace_id: str,
    task_id: str = None,
    field: str = None,
    new_value: str = None
) -> Tuple[str, bool, Any]:
    """Updates a task.

    Parameters:
        workspace_id: Current workspace ID
        task_id: ID of the task
        field: Field to update
        new_value: New value for the field

    Returns:
        (new_workspace_id, success, result)
    """
    action = f'project_management.update_task.func(task_id="{task_id}", field="{field}", new_value="{new_value}")'
    return _execute_tool(workspace_id, action)


# ============================================================================
# CUSTOMER RELATIONSHIP MANAGER TOOLS (5)
# ============================================================================

def customer_relationship_manager_search_customers(
    workspace_id: str,
    customer_name: str = None,
    customer_email: str = None,
    product_interest: str = None,
    status: str = None,
    assigned_to_email: str = None
) -> Tuple[str, bool, Any]:
    """Searches for customers.

    Parameters:
        workspace_id: Current workspace ID
        customer_name: Name of the customer
        customer_email: Email address
        product_interest: Product interest
        status: Status (Qualified, Won, Lost, Lead, Proposal)
        assigned_to_email: Assigned to email

    Returns:
        (new_workspace_id, success, result)
    """
    params = []
    if customer_name:
        params.append(f'customer_name="{customer_name}"')
    if customer_email:
        params.append(f'customer_email="{customer_email}"')
    if product_interest:
        params.append(f'product_interest="{product_interest}"')
    if status:
        params.append(f'status="{status}"')
    if assigned_to_email:
        params.append(f'assigned_to_email="{assigned_to_email}"')

    action = f'customer_relationship_manager.search_customers.func({", ".join(params)})'
    return _execute_tool(workspace_id, action)


def customer_relationship_manager_update_customer(
    workspace_id: str,
    customer_id: str = None,
    field: str = None,
    new_value: str = None
) -> Tuple[str, bool, Any]:
    """Updates a customer.

    Parameters:
        workspace_id: Current workspace ID
        customer_id: ID of the customer
        field: Field to update
        new_value: New value for the field

    Returns:
        (new_workspace_id, success, result)
    """
    action = f'customer_relationship_manager.update_customer.func(customer_id="{customer_id}", field="{field}", new_value="{new_value}")'
    return _execute_tool(workspace_id, action)


def customer_relationship_manager_add_customer(
    workspace_id: str,
    customer_name: str = None,
    assigned_to_email: str = None,
    status: str = None,
    customer_email: str = None,
    customer_phone: str = None,
    last_contact_date: str = None,
    product_interest: str = None,
    notes: str = "",
    follow_up_by: str = None
) -> Tuple[str, bool, Any]:
    """Adds a new customer.

    Parameters:
        workspace_id: Current workspace ID
        customer_name: Name of the customer
        assigned_to_email: Assigned to email
        status: Status (Required: Qualified, Won, Lost, Lead, Proposal)
        customer_email: Customer email address
        customer_phone: Customer phone number
        last_contact_date: Last contact date (YYYY-MM-DD)
        product_interest: Product interest (Software, Hardware, Services, Consulting, Training)
        notes: Notes about customer
        follow_up_by: Follow up date (YYYY-MM-DD)

    Returns:
        (new_workspace_id, success, result)
    """
    # Build params list conditionally to handle None values properly
    params = []
    if customer_name is not None:
        params.append(f'customer_name="{customer_name}"')
    if assigned_to_email is not None:
        params.append(f'assigned_to_email="{assigned_to_email}"')
    if status is not None:
        params.append(f'status="{status}"')
    if customer_email is not None:
        params.append(f'customer_email="{customer_email}"')
    if customer_phone is not None:
        params.append(f'customer_phone="{customer_phone}"')
    if last_contact_date is not None:
        params.append(f'last_contact_date="{last_contact_date}"')
    if product_interest is not None:
        params.append(f'product_interest="{product_interest}"')
    if notes:  # notes has default "", only include if non-empty
        params.append(f'notes="{notes}"')
    if follow_up_by is not None:
        params.append(f'follow_up_by="{follow_up_by}"')

    action = f'customer_relationship_manager.add_customer.func({", ".join(params)})'
    return _execute_tool(workspace_id, action)


def customer_relationship_manager_delete_customer(
    workspace_id: str,
    customer_id: str = None
) -> Tuple[str, bool, Any]:
    """Deletes a customer.

    Parameters:
        workspace_id: Current workspace ID
        customer_id: ID of the customer

    Returns:
        (new_workspace_id, success, result)
    """
    action = f'customer_relationship_manager.delete_customer.func(customer_id="{customer_id}")'
    return _execute_tool(workspace_id, action)


# ============================================================================
# COMPANY DIRECTORY TOOLS (1)
# ============================================================================

def company_directory_find_email_address(
    workspace_id: str,
    name: str = ""
) -> Tuple[str, bool, Any]:
    """Finds email address by name.

    Parameters:
        workspace_id: Current workspace ID
        name: Name to search for

    Returns:
        (new_workspace_id, success, result)
    """
    action = f'company_directory.find_email_address.func(name="{name}")'
    return _execute_tool(workspace_id, action)


# ============================================================================
# TOOL REGISTRY
# ============================================================================

ALL_TOOLS = {
    # Calendar
    "calendar.get_event_information_by_id": calendar_get_event_information_by_id,
    "calendar.search_events": calendar_search_events,
    "calendar.create_event": calendar_create_event,
    "calendar.delete_event": calendar_delete_event,
    "calendar.update_event": calendar_update_event,

    # Email
    "email.get_email_information_by_id": email_get_email_information_by_id,
    "email.search_emails": email_search_emails,
    "email.send_email": email_send_email,
    "email.delete_email": email_delete_email,
    "email.forward_email": email_forward_email,
    "email.reply_email": email_reply_email,

    # Analytics
    "analytics.get_visitor_information_by_id": analytics_get_visitor_information_by_id,
    "analytics.create_plot": analytics_create_plot,
    "analytics.total_visits_count": analytics_total_visits_count,
    "analytics.engaged_users_count": analytics_engaged_users_count,
    "analytics.traffic_source_count": analytics_traffic_source_count,
    "analytics.get_average_session_duration": analytics_get_average_session_duration,

    # Project Management
    "project_management.get_task_information_by_id": project_management_get_task_information_by_id,
    "project_management.search_tasks": project_management_search_tasks,
    "project_management.create_task": project_management_create_task,
    "project_management.delete_task": project_management_delete_task,
    "project_management.update_task": project_management_update_task,

    # Customer Relationship Manager
    "customer_relationship_manager.search_customers": customer_relationship_manager_search_customers,
    "customer_relationship_manager.update_customer": customer_relationship_manager_update_customer,
    "customer_relationship_manager.add_customer": customer_relationship_manager_add_customer,
    "customer_relationship_manager.delete_customer": customer_relationship_manager_delete_customer,

    # Company Directory
    "company_directory.find_email_address": company_directory_find_email_address,
}
