# WorkBench Core Module

File-system based workspace management with 34 tool functions for multi-agent beam search.

## Quick Start

```python
from core.workspace_manager import WorkspaceManager
from core import tools

# Initialize
manager = WorkspaceManager("_temp_workspace")
tools.initialize_tools(manager)

# Create workspace
ws_id = manager.create_workspace("Find all meetings this week")

# Execute tool
new_ws_id, success, result = tools.calendar_search_events(
    ws_id,
    query="meeting",
    time_min="2023-11-20",
    time_max="2023-11-27"
)

# Clean up
manager.delete_workspace(ws_id)
manager.delete_workspace(new_ws_id)
```

## Core API

### WorkspaceManager

| Command | Description | Returns |
|---------|-------------|---------|
| `__init__(base_path, template_base_path)` | Initialize manager | Manager instance |
| `create_workspace(query)` | Create new workspace | `workspace_id` |
| `delete_workspace(workspace_id)` | Delete workspace | None |
| `fork_and_execute(source_id, action, executor_func)` | Fork & execute | `(new_id, success, result)` |
| `get_workspace_path(workspace_id)` | Get workspace path | `Path` object |

### Tools Module

| Command | Description |
|---------|-------------|
| `initialize_tools(manager)` | Initialize tools (required) |

**All tool functions follow this pattern:**
- Input: `workspace_id` + parameters
- Output: `(new_workspace_id, success, result)`

---

## API Reference

### 1. WorkspaceManager Commands

#### `create_workspace(query="")`
Create initial workspace with CSV files.

**Example:**
```python
ws_id = manager.create_workspace("Analyze sales data")
# Returns: "20260204_a3b5c2"
```

#### `delete_workspace(workspace_id)`
Delete workspace and contents.

**Example:**
```python
manager.delete_workspace("20260204_a3b5c2")
```

#### `fork_and_execute(source_id, action, executor_func)`
Fork workspace, execute action, log result.

**Example:**
```python
from core.tool_executor import execute_action

new_id, success, result = manager.fork_and_execute(
    "20260204_a3b5c2",
    'calendar.search_events.func(query="meeting")',
    execute_action
)
```

#### `get_workspace_path(workspace_id)`
Get filesystem path.

**Example:**
```python
path = manager.get_workspace_path("20260204_a3b5c2")
# Returns: Path("/path/to/_temp_workspace/20260204_a3b5c2")
```

---

### 2. Calendar Tools (5)

#### `calendar_search_events(workspace_id, query, time_min, time_max)`
```python
ws, success, events = tools.calendar_search_events(
    ws_id, query="meeting",
    time_min="2023-11-20 00:00:00",
    time_max="2023-11-30 23:59:59"
)
```

#### `calendar_create_event(workspace_id, event_name, participant_email, event_start, duration)`
```python
ws, success, event_id = tools.calendar_create_event(
    ws_id,
    event_name="Team Meeting",
    participant_email="john@example.com",
    event_start="2023-12-01 14:00:00",
    duration="60"
)
```

#### `calendar_get_event_information_by_id(workspace_id, event_id, field)`
```python
ws, success, info = tools.calendar_get_event_information_by_id(
    ws_id, event_id="00000001", field="event_name"
)
```

#### `calendar_update_event(workspace_id, event_id, field, new_value)`
```python
ws, success, msg = tools.calendar_update_event(
    ws_id, event_id="00000001",
    field="event_name", new_value="Updated Name"
)
```

#### `calendar_delete_event(workspace_id, event_id)`
```python
ws, success, msg = tools.calendar_delete_event(
    ws_id, event_id="00000001"
)
```

---

### 3. Email Tools (6)

#### `email_search_emails(workspace_id, query, date_min, date_max)`
```python
ws, success, emails = tools.email_search_emails(
    ws_id, query="project",
    date_min="2023-11-01",
    date_max="2023-11-30"
)
```

#### `email_send_email(workspace_id, recipient, subject, body)`
```python
ws, success, email_id = tools.email_send_email(
    ws_id,
    recipient="john@example.com",
    subject="Project Update",
    body="The project is on track."
)
```

#### `email_get_email_information_by_id(workspace_id, email_id, field)`
```python
ws, success, info = tools.email_get_email_information_by_id(
    ws_id, email_id="email_001", field="subject"
)
```

#### `email_forward_email(workspace_id, email_id, recipient)`
```python
ws, success, new_email_id = tools.email_forward_email(
    ws_id, email_id="email_001",
    recipient="jane@example.com"
)
```

#### `email_reply_email(workspace_id, email_id, body)`
```python
ws, success, reply_id = tools.email_reply_email(
    ws_id, email_id="email_001",
    body="Thanks for the update!"
)
```

#### `email_delete_email(workspace_id, email_id)`
```python
ws, success, msg = tools.email_delete_email(
    ws_id, email_id="email_001"
)
```

---

### 4. Analytics Tools (6)

#### `analytics_create_plot(workspace_id, time_min, time_max, value_to_plot, plot_type)`
```python
ws, success, path = tools.analytics_create_plot(
    ws_id,
    time_min="2023-11-21",
    time_max="2023-11-29",
    value_to_plot="total_visits",
    plot_type="bar"
)
# path: "plots/2023-11-21_2023-11-29_total_visits_bar.png"
```

#### `analytics_total_visits_count(workspace_id, time_min, time_max)`
```python
ws, success, counts = tools.analytics_total_visits_count(
    ws_id, time_min="2023-11-21", time_max="2023-11-29"
)
# counts: {"2023-11-21": 11, "2023-11-22": 10, ...}
```

#### `analytics_engaged_users_count(workspace_id, time_min, time_max)`
```python
ws, success, counts = tools.analytics_engaged_users_count(
    ws_id, time_min="2023-11-21", time_max="2023-11-29"
)
```

#### `analytics_traffic_source_count(workspace_id, time_min, time_max, traffic_source)`
```python
ws, success, count = tools.analytics_traffic_source_count(
    ws_id, time_min="2023-11-21",
    time_max="2023-11-29",
    traffic_source="organic"
)
```

#### `analytics_get_average_session_duration(workspace_id, time_min, time_max)`
```python
ws, success, avg = tools.analytics_get_average_session_duration(
    ws_id, time_min="2023-11-21", time_max="2023-11-29"
)
```

#### `analytics_get_visitor_information_by_id(workspace_id, visitor_id)`
```python
ws, success, info = tools.analytics_get_visitor_information_by_id(
    ws_id, visitor_id="visitor_001"
)
```

---

### 5. Project Management Tools (5)

#### `project_management_search_tasks(workspace_id, task_name, assigned_to_email, list_name, due_date, board)`
```python
ws, success, tasks = tools.project_management_search_tasks(
    ws_id, task_name="review", board="Development"
)
```

#### `project_management_create_task(workspace_id, task_name, assigned_to_email, list_name, due_date, board)`
```python
ws, success, task_id = tools.project_management_create_task(
    ws_id,
    task_name="Fix Bug #123",
    assigned_to_email="john@example.com",
    list_name="In Progress",
    due_date="2023-12-15",
    board="Development"
)
```

#### `project_management_get_task_information_by_id(workspace_id, task_id, field)`
```python
ws, success, info = tools.project_management_get_task_information_by_id(
    ws_id, task_id="task_001", field="task_name"
)
```

#### `project_management_update_task(workspace_id, task_id, field, new_value)`
```python
ws, success, msg = tools.project_management_update_task(
    ws_id, task_id="task_001",
    field="list_name", new_value="Done"
)
```

#### `project_management_delete_task(workspace_id, task_id)`
```python
ws, success, msg = tools.project_management_delete_task(
    ws_id, task_id="task_001"
)
```

---

### 6. Customer Relationship Manager Tools (4)

#### `customer_relationship_manager_search_customers(workspace_id, customer_name, email, phone, company)`
```python
ws, success, customers = tools.customer_relationship_manager_search_customers(
    ws_id, customer_name="John", company="Acme Corp"
)
```

#### `customer_relationship_manager_add_customer(workspace_id, customer_name, email, phone, company, account_value, last_contact_date, interaction_count)`
```python
ws, success, cust_id = tools.customer_relationship_manager_add_customer(
    ws_id,
    customer_name="Jane Smith",
    email="jane@example.com",
    phone="555-1234",
    company="Tech Inc",
    account_value="75000",
    last_contact_date="2023-11-20",
    interaction_count="5"
)
```

#### `customer_relationship_manager_update_customer(workspace_id, customer_id, field, new_value)`
```python
ws, success, msg = tools.customer_relationship_manager_update_customer(
    ws_id, customer_id="cust_001",
    field="account_value", new_value="50000"
)
```

#### `customer_relationship_manager_delete_customer(workspace_id, customer_id)`
```python
ws, success, msg = tools.customer_relationship_manager_delete_customer(
    ws_id, customer_id="cust_001"
)
```

---

### 7. Company Directory Tools (1)

#### `company_directory_find_email_address(workspace_id, name)`
```python
ws, success, emails = tools.company_directory_find_email_address(
    ws_id, name="john"
)
# emails: ["john.doe@company.com", "john.smith@company.com"]
```

---

## Complete Workflow Example

```python
from core.workspace_manager import WorkspaceManager
from core import tools

# Initialize
manager = WorkspaceManager()
tools.initialize_tools(manager)

# Create workspace
ws = manager.create_workspace("Monthly report")

# Search meetings
ws, ok, meetings = tools.calendar_search_events(
    ws, query="meeting",
    time_min="2023-11-01 00:00:00",
    time_max="2023-11-30 23:59:59"
)
print(f"Meetings: {len(meetings)}")

# Get analytics
ws, ok, visits = tools.analytics_total_visits_count(
    ws, time_min="2023-11-01", time_max="2023-11-30"
)
print(f"Total visits: {sum(visits.values())}")

# Create plot
ws, ok, plot = tools.analytics_create_plot(
    ws, time_min="2023-11-01", time_max="2023-11-30",
    value_to_plot="total_visits", plot_type="line"
)
print(f"Plot: {plot}")

# Send email
ws, ok, email_id = tools.email_send_email(
    ws,
    recipient="manager@example.com",
    subject="Monthly Report",
    body=f"Meetings: {len(meetings)}, Visits: {sum(visits.values())}"
)
print(f"Email sent: {email_id}")

# Cleanup
manager.delete_workspace(ws)
```

---

## Testing

Run comprehensive test suite:
```bash
python test_all_features.py
```
