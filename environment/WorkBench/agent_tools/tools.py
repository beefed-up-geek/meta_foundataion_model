"""LangChain Tool Wrappers for WorkBench

These are tool specifications for LLM agents. Each tool is a placeholder that
will be connected to core.tools functions during execution.

Note: These tools do NOT include workspace_id parameter - that will be managed
by the execution framework.

================================================================================
IMPORTANT: Integration with core.tools for Evaluation
================================================================================

이 파일의 도구들은 LLM Agent에게 도구 스펙(이름, 설명, 파라미터)을 제공하는 placeholder입니다.
실제 실행 로직은 core/tools.py에 구현되어 있습니다.

Agent 평가 환경에서 사용할 때는 다음과 같이 연동해야 합니다:

1. WorkspaceManager로 workspace 생성
2. 이 파일의 도구를 Agent에 제공 (LLM이 도구 호출 방법을 학습)
3. Agent가 도구를 호출하면, 해당 호출을 가로채서 core.tools의 실제 함수 실행
4. workspace_id는 실행 프레임워크가 관리

예시 (workbench_solver.py 참조):
```python
from environment.WorkBench.agent_tools.tools import ALL_TOOLS, TOOL_MAPPING
from environment.WorkBench.core import tools as core_tools

# workspace_id를 클로저로 캡처하여 실제 도구 생성
def create_connected_tools(workspace_id: str):
    connected_tools = []
    for tool in ALL_TOOLS:
        core_func_name = TOOL_MAPPING[tool.name]  # e.g., "core.tools.calendar_search_events"
        core_func = getattr(core_tools, core_func_name.split('.')[-1])
        # workspace_id를 바인딩한 새 도구 생성
        ...
    return connected_tools
```

TOOL_MAPPING 딕셔너리는 이 파일 하단에 정의되어 있습니다.
================================================================================
"""

from langchain.tools import tool


# ============================================================================
# CALENDAR TOOLS (5)
# ============================================================================

# Connect to: core.tools.calendar_search_events
@tool("calendar_search_events")
def calendar_search_events(query: str = "", time_min: str = None, time_max: str = None):
    """Search for calendar events by query and time range.

    Use this tool to find events that match a search query or fall within a time period.

    Args:
        query: Search term to match in event names and participant emails (e.g., "meeting", "team")
        time_min: Start of time range in format "YYYY-MM-DD HH:MM:SS" (optional)
        time_max: End of time range in format "YYYY-MM-DD HH:MM:SS" (optional)

    Returns:
        List of matching events (up to 5), each containing event_id, event_name,
        participant_email, event_start, and duration

    Examples:
        - Search for meetings: calendar_search_events(query="meeting")
        - Find events in November: calendar_search_events(time_min="2023-11-01 00:00:00", time_max="2023-11-30 23:59:59")
    """
    pass


# Connect to: core.tools.calendar_create_event
@tool("calendar_create_event")
def calendar_create_event(
    event_name: str,
    participant_email: str,
    event_start: str,
    duration: str
):
    """Create a new calendar event.

    Use this tool to schedule a new event with a participant.

    Args:
        event_name: Name/title of the event (required)
        participant_email: Email address of the participant (required)
        event_start: Start time in format "YYYY-MM-DD HH:MM:SS" (required)
        duration: Duration in minutes as a string (required, e.g., "60" for 1 hour)

    Returns:
        Event ID of the newly created event (8-digit string)

    Example:
        - Schedule a meeting: calendar_create_event(
            event_name="Team Standup",
            participant_email="john@example.com",
            event_start="2023-12-01 09:00:00",
            duration="30"
          )
    """
    pass


# Connect to: core.tools.calendar_get_event_information_by_id
@tool("calendar_get_event_information_by_id")
def calendar_get_event_information_by_id(event_id: str, field: str):
    """Get specific information about a calendar event by its ID.

    Use this tool to retrieve a specific field from an event.

    Args:
        event_id: 8-digit event ID (required)
        field: Field name to retrieve (required)
               Options: "event_id", "event_name", "participant_email", "event_start", "duration"

    Returns:
        Dictionary with the requested field and its value

    Example:
        - Get event name: calendar_get_event_information_by_id(event_id="00000123", field="event_name")
    """
    pass


# Connect to: core.tools.calendar_update_event
@tool("calendar_update_event")
def calendar_update_event(event_id: str, field: str, new_value: str):
    """Update a field of an existing calendar event.

    Use this tool to modify event details.

    Args:
        event_id: 8-digit event ID (required)
        field: Field name to update (required)
               Options: "event_name", "participant_email", "event_start", "duration"
        new_value: New value for the field (required)

    Returns:
        Success message

    Example:
        - Change event time: calendar_update_event(
            event_id="00000123",
            field="event_start",
            new_value="2023-12-01 14:00:00"
          )
    """
    pass


# Connect to: core.tools.calendar_delete_event
@tool("calendar_delete_event")
def calendar_delete_event(event_id: str):
    """Delete a calendar event by its ID.

    Use this tool to remove an event from the calendar.

    Args:
        event_id: 8-digit event ID (required)

    Returns:
        Success message

    Example:
        - Delete event: calendar_delete_event(event_id="00000123")
    """
    pass


# ============================================================================
# EMAIL TOOLS (6)
# ============================================================================

# Connect to: core.tools.email_search_emails
@tool("email_search_emails")
def email_search_emails(query: str = "", date_min: str = None, date_max: str = None):
    """Search for emails by query and date range.

    Use this tool to find emails matching search criteria.

    Args:
        query: Search term to match in subject, sender, recipient, or body (optional)
        date_min: Earliest date in format "YYYY-MM-DD" (optional)
        date_max: Latest date in format "YYYY-MM-DD" (optional)

    Returns:
        List of matching emails (up to 5), each containing email_id, sender,
        recipient, subject, body, and timestamp

    Example:
        - Search for project emails: email_search_emails(query="project")
        - Find emails from November: email_search_emails(date_min="2023-11-01", date_max="2023-11-30")
    """
    pass


# Connect to: core.tools.email_send_email
@tool("email_send_email")
def email_send_email(recipient: str, subject: str, body: str):
    """Send a new email.

    Use this tool to compose and send an email.

    Args:
        recipient: Email address of the recipient (required)
        subject: Email subject line (required)
        body: Email body content (required)

    Returns:
        Email ID of the sent email

    Example:
        - Send status update: email_send_email(
            recipient="manager@example.com",
            subject="Project Status Update",
            body="The project is on track for delivery."
          )
    """
    pass


# Connect to: core.tools.email_get_email_information_by_id
@tool("email_get_email_information_by_id")
def email_get_email_information_by_id(email_id: str, field: str):
    """Get specific information about an email by its ID.

    Use this tool to retrieve a specific field from an email.

    Args:
        email_id: Email ID (required)
        field: Field name to retrieve (required)
               Options: "email_id", "sender", "recipient", "subject", "body", "timestamp"

    Returns:
        Dictionary with the requested field and its value

    Example:
        - Get email subject: email_get_email_information_by_id(email_id="email_001", field="subject")
    """
    pass


# Connect to: core.tools.email_forward_email
@tool("email_forward_email")
def email_forward_email(email_id: str, recipient: str):
    """Forward an existing email to another recipient.

    Use this tool to forward an email.

    Args:
        email_id: ID of the email to forward (required)
        recipient: Email address to forward to (required)

    Returns:
        Email ID of the forwarded email

    Example:
        - Forward to colleague: email_forward_email(email_id="email_001", recipient="john@example.com")
    """
    pass


# Connect to: core.tools.email_reply_email
@tool("email_reply_email")
def email_reply_email(email_id: str, body: str):
    """Reply to an existing email.

    Use this tool to send a reply to an email.

    Args:
        email_id: ID of the email to reply to (required)
        body: Reply message content (required)

    Returns:
        Email ID of the reply

    Example:
        - Reply with confirmation: email_reply_email(email_id="email_001", body="Confirmed, I'll attend.")
    """
    pass


# Connect to: core.tools.email_delete_email
@tool("email_delete_email")
def email_delete_email(email_id: str):
    """Delete an email by its ID.

    Use this tool to remove an email.

    Args:
        email_id: Email ID (required)

    Returns:
        Success message

    Example:
        - Delete spam: email_delete_email(email_id="email_001")
    """
    pass


# ============================================================================
# ANALYTICS TOOLS (6)
# ============================================================================

# Connect to: core.tools.analytics_create_plot
@tool("analytics_create_plot")
def analytics_create_plot(
    time_min: str,
    time_max: str,
    value_to_plot: str,
    plot_type: str
):
    """Create a visualization plot of analytics data.

    Use this tool to generate charts for data analysis.

    CRITICAL DATE RULE:
    - Today is 2023-11-30, but today's analytics data is NOT yet complete.
    - For queries like "since X" or "from X", use YESTERDAY (2023-11-29) as time_max, NOT today.
    - Example: "since November 21" → time_min="2023-11-21", time_max="2023-11-29"

    Args:
        time_min: Start date in format "YYYY-MM-DD" (required)
        time_max: End date in format "YYYY-MM-DD" (required) - Use 2023-11-29 for "until now" queries
        value_to_plot: Metric to visualize (required)
                       Options: "total_visits", "user_engaged"
        plot_type: Type of chart (required)
                   Options: "bar", "line", "scatter", "histogram"

    Returns:
        File path to the generated plot image

    Example:
        - Create visits bar chart since Nov 21: analytics_create_plot(
            time_min="2023-11-21",
            time_max="2023-11-29",
            value_to_plot="total_visits",
            plot_type="bar"
          )
    """
    pass


# Connect to: core.tools.analytics_total_visits_count
@tool("analytics_total_visits_count")
def analytics_total_visits_count(time_min: str = None, time_max: str = None):
    """Get total visit counts by date.

    Use this tool to retrieve visitor traffic statistics.

    Args:
        time_min: Start date in format "YYYY-MM-DD" (optional)
        time_max: End date in format "YYYY-MM-DD" (optional)

    Returns:
        Dictionary mapping dates to visit counts

    Example:
        - Get November visits: analytics_total_visits_count(time_min="2023-11-01", time_max="2023-11-30")
    """
    pass


# Connect to: core.tools.analytics_engaged_users_count
@tool("analytics_engaged_users_count")
def analytics_engaged_users_count(time_min: str = None, time_max: str = None):
    """Get engaged user counts by date.

    Use this tool to measure user engagement.

    Args:
        time_min: Start date in format "YYYY-MM-DD" (optional)
        time_max: End date in format "YYYY-MM-DD" (optional)

    Returns:
        Dictionary mapping dates to engaged user counts

    Example:
        - Get engagement data: analytics_engaged_users_count(time_min="2023-11-01", time_max="2023-11-30")
    """
    pass


# Connect to: core.tools.analytics_traffic_source_count
@tool("analytics_traffic_source_count")
def analytics_traffic_source_count(
    time_min: str = None,
    time_max: str = None,
    traffic_source: str = None
):
    """Get visitor count by traffic source.

    Use this tool to analyze where visitors come from.

    Args:
        time_min: Start date in format "YYYY-MM-DD" (optional)
        time_max: End date in format "YYYY-MM-DD" (optional)
        traffic_source: Source to filter by (optional)
                        Examples: "organic", "direct", "referral", "social"

    Returns:
        Count of visitors from the specified source

    Example:
        - Get organic traffic: analytics_traffic_source_count(
            time_min="2023-11-01",
            time_max="2023-11-30",
            traffic_source="organic"
          )
    """
    pass


# Connect to: core.tools.analytics_get_average_session_duration
@tool("analytics_get_average_session_duration")
def analytics_get_average_session_duration(time_min: str = None, time_max: str = None):
    """Get average session duration for visitors.

    Use this tool to measure user engagement time.

    Args:
        time_min: Start date in format "YYYY-MM-DD" (optional)
        time_max: End date in format "YYYY-MM-DD" (optional)

    Returns:
        Average session duration in seconds as a string

    Example:
        - Get average session time: analytics_get_average_session_duration(
            time_min="2023-11-01",
            time_max="2023-11-30"
          )
    """
    pass


# Connect to: core.tools.analytics_get_visitor_information_by_id
@tool("analytics_get_visitor_information_by_id")
def analytics_get_visitor_information_by_id(visitor_id: str):
    """Get detailed information about a specific visitor.

    Use this tool to look up visitor details.

    Args:
        visitor_id: Visitor ID (required)

    Returns:
        Dictionary with visitor information including visitor_id, timestamp,
        session_duration, and traffic_source

    Example:
        - Get visitor details: analytics_get_visitor_information_by_id(visitor_id="00000001")
    """
    pass


# ============================================================================
# PROJECT MANAGEMENT TOOLS (5)
# ============================================================================

# Connect to: core.tools.project_management_search_tasks
@tool("project_management_search_tasks")
def project_management_search_tasks(
    task_name: str = None,
    assigned_to_email: str = None,
    list_name: str = None,
    due_date: str = None,
    board: str = None
):
    """Search for tasks by various criteria.

    Use this tool to find tasks matching search parameters.

    Args:
        task_name: Task name to search for (optional)
        assigned_to_email: Email of assigned person (optional)
        list_name: List/column name (optional, e.g., "To Do", "In Progress", "Done")
        due_date: Due date in format "YYYY-MM-DD" (optional)
        board: Board name (optional, e.g., "Development", "Marketing")

    Returns:
        List of matching tasks (up to 5), each containing task_id, task_name,
        assigned_to_email, list_name, due_date, and board

    Example:
        - Find review tasks: project_management_search_tasks(task_name="review")
        - Find urgent tasks: project_management_search_tasks(list_name="In Progress", board="Development")
    """
    pass


# Connect to: core.tools.project_management_create_task
@tool("project_management_create_task")
def project_management_create_task(
    task_name: str,
    assigned_to_email: str,
    list_name: str,
    due_date: str,
    board: str
):
    """Create a new task.

    Use this tool to add a task to a project board.

    Args:
        task_name: Name/description of the task (required)
        assigned_to_email: Email of person to assign (required)
        list_name: List/column (required, e.g., "To Do", "In Progress")
        due_date: Due date in format "YYYY-MM-DD" (required)
        board: Board name (required, e.g., "Development")

    Returns:
        Task ID of the newly created task

    Example:
        - Create task: project_management_create_task(
            task_name="Implement login feature",
            assigned_to_email="john@example.com",
            list_name="To Do",
            due_date="2023-12-15",
            board="Development"
          )
    """
    pass


# Connect to: core.tools.project_management_get_task_information_by_id
@tool("project_management_get_task_information_by_id")
def project_management_get_task_information_by_id(task_id: str, field: str):
    """Get specific information about a task by its ID.

    Use this tool to retrieve a specific field from a task.

    Args:
        task_id: Task ID (required)
        field: Field name to retrieve (required)
               Options: "task_id", "task_name", "assigned_to_email", "list_name", "due_date", "board"

    Returns:
        Dictionary with the requested field and its value

    Example:
        - Get task status: project_management_get_task_information_by_id(task_id="task_001", field="list_name")
    """
    pass


# Connect to: core.tools.project_management_update_task
@tool("project_management_update_task")
def project_management_update_task(task_id: str, field: str, new_value: str):
    """Update a field of an existing task.

    Use this tool to modify task details.

    Args:
        task_id: Task ID (required)
        field: Field name to update (required)
               Options: "task_name", "assigned_to_email", "list_name", "due_date", "board"
        new_value: New value for the field (required)

    Returns:
        Success message

    Example:
        - Move to done: project_management_update_task(task_id="task_001", field="list_name", new_value="Done")
    """
    pass


# Connect to: core.tools.project_management_delete_task
@tool("project_management_delete_task")
def project_management_delete_task(task_id: str):
    """Delete a task by its ID.

    Use this tool to remove a task from the board.

    Args:
        task_id: Task ID (required)

    Returns:
        Success message

    Example:
        - Delete task: project_management_delete_task(task_id="task_001")
    """
    pass


# ============================================================================
# CUSTOMER RELATIONSHIP MANAGER TOOLS (4)
# ============================================================================

# Connect to: core.tools.customer_relationship_manager_search_customers
@tool("customer_relationship_manager_search_customers")
def customer_relationship_manager_search_customers(
    customer_name: str = None,
    customer_email: str = None,
    product_interest: str = None,
    status: str = None,
    assigned_to_email: str = None
):
    """Search for customers by various criteria.

    Use this tool to find customers matching search parameters.

    Args:
        customer_name: Customer name to search for (optional)
        customer_email: Customer email address (optional)
        product_interest: Product interest (optional)
                          Options: "Software", "Hardware", "Services", "Consulting", "Training"
        status: Customer status (optional)
                Options: "Qualified", "Won", "Lost", "Lead", "Proposal"
        assigned_to_email: Email of assigned salesperson (optional)

    Returns:
        List of matching customers (up to 5), each containing customer details

    Example:
        - Find leads: customer_relationship_manager_search_customers(status="Lead")
        - Find software customers: customer_relationship_manager_search_customers(product_interest="Software")
    """
    pass


# Connect to: core.tools.customer_relationship_manager_add_customer
@tool("customer_relationship_manager_add_customer")
def customer_relationship_manager_add_customer(
    customer_name: str,
    assigned_to_email: str,
    status: str,
    customer_email: str = None,
    customer_phone: str = None,
    last_contact_date: str = None,
    product_interest: str = None,
    notes: str = "",
    follow_up_by: str = None
):
    """Add a new customer to the CRM.

    Use this tool to create a customer record.

    Args:
        customer_name: Name of the customer (required)
        assigned_to_email: Email of assigned salesperson (required)
        status: Customer status (required)
                Options: "Qualified", "Won", "Lost", "Lead", "Proposal"
        customer_email: Customer's email address (optional)
        customer_phone: Customer's phone number (optional)
        last_contact_date: Last contact date in format "YYYY-MM-DD" (optional)
        product_interest: Product interest (optional)
                          Options: "Software", "Hardware", "Services", "Consulting", "Training"
        notes: Additional notes (optional)
        follow_up_by: Follow-up date in format "YYYY-MM-DD" (optional)

    Returns:
        Customer ID of the newly created customer

    Example:
        - Add new lead: customer_relationship_manager_add_customer(
            customer_name="Acme Corp",
            assigned_to_email="sales@example.com",
            status="Lead",
            customer_email="contact@acme.com",
            product_interest="Software"
          )
    """
    pass


# Connect to: core.tools.customer_relationship_manager_update_customer
@tool("customer_relationship_manager_update_customer")
def customer_relationship_manager_update_customer(
    customer_id: str,
    field: str,
    new_value: str
):
    """Update a field of an existing customer record.

    Use this tool to modify customer details.

    Args:
        customer_id: Customer ID (required)
        field: Field name to update (required)
               Options: "customer_name", "assigned_to_email", "customer_email", "customer_phone",
                        "last_contact_date", "product_interest", "status", "notes", "follow_up_by"
        new_value: New value for the field (required)

    Returns:
        Success message

    Example:
        - Update status: customer_relationship_manager_update_customer(
            customer_id="cust_001",
            field="status",
            new_value="Won"
          )
    """
    pass


# Connect to: core.tools.customer_relationship_manager_delete_customer
@tool("customer_relationship_manager_delete_customer")
def customer_relationship_manager_delete_customer(customer_id: str):
    """Delete a customer record by its ID.

    Use this tool to remove a customer from the CRM.

    Args:
        customer_id: Customer ID (required)

    Returns:
        Success message

    Example:
        - Delete customer: customer_relationship_manager_delete_customer(customer_id="cust_001")
    """
    pass


# ============================================================================
# COMPANY DIRECTORY TOOLS (1)
# ============================================================================

# Connect to: core.tools.company_directory_find_email_address
@tool("company_directory_find_email_address")
def company_directory_find_email_address(name: str):
    """Find email addresses by employee name.

    Use this tool to look up employee email addresses in the company directory.

    Args:
        name: Name to search for (required)
              Can be partial (e.g., "john" will match "John Doe", "John Smith")

    Returns:
        List of matching email addresses

    Example:
        - Find John's email: company_directory_find_email_address(name="john")
        - Find manager: company_directory_find_email_address(name="manager")
    """
    pass


# ============================================================================
# TOOL REGISTRY
# ============================================================================

ALL_TOOLS = [
    # Calendar (5)
    calendar_search_events,
    calendar_create_event,
    calendar_get_event_information_by_id,
    calendar_update_event,
    calendar_delete_event,

    # Email (6)
    email_search_emails,
    email_send_email,
    email_get_email_information_by_id,
    email_forward_email,
    email_reply_email,
    email_delete_email,

    # Analytics (6)
    analytics_create_plot,
    analytics_total_visits_count,
    analytics_engaged_users_count,
    analytics_traffic_source_count,
    analytics_get_average_session_duration,
    analytics_get_visitor_information_by_id,

    # Project Management (5)
    project_management_search_tasks,
    project_management_create_task,
    project_management_get_task_information_by_id,
    project_management_update_task,
    project_management_delete_task,

    # CRM (4)
    customer_relationship_manager_search_customers,
    customer_relationship_manager_add_customer,
    customer_relationship_manager_update_customer,
    customer_relationship_manager_delete_customer,

    # Company Directory (1)
    company_directory_find_email_address,
]


# Tool mapping for connecting to core.tools
TOOL_MAPPING = {
    "calendar_search_events": "core.tools.calendar_search_events",
    "calendar_create_event": "core.tools.calendar_create_event",
    "calendar_get_event_information_by_id": "core.tools.calendar_get_event_information_by_id",
    "calendar_update_event": "core.tools.calendar_update_event",
    "calendar_delete_event": "core.tools.calendar_delete_event",

    "email_search_emails": "core.tools.email_search_emails",
    "email_send_email": "core.tools.email_send_email",
    "email_get_email_information_by_id": "core.tools.email_get_email_information_by_id",
    "email_forward_email": "core.tools.email_forward_email",
    "email_reply_email": "core.tools.email_reply_email",
    "email_delete_email": "core.tools.email_delete_email",

    "analytics_create_plot": "core.tools.analytics_create_plot",
    "analytics_total_visits_count": "core.tools.analytics_total_visits_count",
    "analytics_engaged_users_count": "core.tools.analytics_engaged_users_count",
    "analytics_traffic_source_count": "core.tools.analytics_traffic_source_count",
    "analytics_get_average_session_duration": "core.tools.analytics_get_average_session_duration",
    "analytics_get_visitor_information_by_id": "core.tools.analytics_get_visitor_information_by_id",

    "project_management_search_tasks": "core.tools.project_management_search_tasks",
    "project_management_create_task": "core.tools.project_management_create_task",
    "project_management_get_task_information_by_id": "core.tools.project_management_get_task_information_by_id",
    "project_management_update_task": "core.tools.project_management_update_task",
    "project_management_delete_task": "core.tools.project_management_delete_task",

    "customer_relationship_manager_search_customers": "core.tools.customer_relationship_manager_search_customers",
    "customer_relationship_manager_add_customer": "core.tools.customer_relationship_manager_add_customer",
    "customer_relationship_manager_update_customer": "core.tools.customer_relationship_manager_update_customer",
    "customer_relationship_manager_delete_customer": "core.tools.customer_relationship_manager_delete_customer",

    "company_directory_find_email_address": "core.tools.company_directory_find_email_address",
}
