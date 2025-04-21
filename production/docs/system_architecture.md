# VILA-15B Production System Architecture

## System Overview

The VILA-15B production system is designed to autonomously ingest new data and trigger fine-tuning jobs. The architecture focuses on clear component isolation, defined interfaces, and standardized workflows.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        VILA-15B Production System                           │
│                                                                             │
│  ┌───────────┐    ┌────────────┐    ┌────────────┐    ┌──────────────────┐  │
│  │           │    │            │    │            │    │                  │  │
│  │  Data     ├───►│  Data      ├───►│  Fine-     ├───►│  Model           │  │
│  │  Ingestion│    │  Validation│    │  Tuning    │    │  Registration    │  │
│  │           │    │            │    │            │    │                  │  │
│  └───────────┘    └────────────┘    └────────────┘    └──────────────────┘  │
│        │                 │                │                   │              │
│        ▼                 ▼                ▼                   ▼              │
│  ┌───────────┐    ┌────────────┐    ┌────────────┐    ┌──────────────────┐  │
│  │           │    │            │    │            │    │                  │  │
│  │  /data/   │    │  /data/    │    │ Compute    │    │  /models/        │  │
│  │  raw      │    │  processed │    │ Resources  │    │  versioned       │  │
│  │           │    │            │    │            │    │                  │  │
│  └───────────┘    └────────────┘    └────────────┘    └──────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │                     Orchestration Service                          │      │
│  │                                                                    │      │
│  │  ┌───────────┐    ┌────────────┐    ┌────────────────────┐        │      │
│  │  │           │    │            │    │                    │        │      │
│  │  │  Event    │    │  Job       │    │  Model Evaluation  │        │      │
│  │  │  System   │    │  Scheduler │    │  & Promotion       │        │      │
│  │  │           │    │            │    │                    │        │      │
│  │  └───────────┘    └────────────┘    └────────────────────┘        │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Interfaces

### 1. Data Ingestion → Data Validation

**Purpose**: Transfer raw input data to validation service

**Interface Type**: File system + Event notification

**Contract**:
- Data Ingestion places new files in `/data/raw/{dataset_name}/`
- Metadata file `dataset_info.json` provides context about the dataset
- Triggers event `dataset.new` with payload containing dataset path
- Data Validation acknowledges receipt via event `dataset.received`

**Example Payload**:
```json
{
  "dataset_name": "NW-test-camera-2",
  "dataset_path": "/data/raw/NW-test-camera-2",
  "timestamp": "2025-04-21T15:30:00Z",
  "file_count": 243
}
```

### 2. Data Validation → Fine-Tuning

**Purpose**: Provide validated data for fine-tuning

**Interface Type**: File system + Event notification

**Contract**:
- Data Validation processes raw data and places validated datasets in `/data/processed/{dataset_name}/`
- Creates `validation_report.json` with validation details and statistics
- Triggers event `dataset.validated` when processing complete
- Fine-Tuning service listens for this event to initiate training

**Example Payload**:
```json
{
  "dataset_name": "NW-test-camera-2",
  "dataset_path": "/data/processed/NW-test-camera-2",
  "validation_status": "PASSED",
  "sample_count": 240,
  "timestamp": "2025-04-21T15:45:00Z"
}
```

### 3. Fine-Tuning → Model Registration

**Purpose**: Register newly trained models for versioning and serving

**Interface Type**: File system + Registration API

**Contract**:
- Fine-Tuning service produces model artifacts in temporary location
- Calls Registration API with model location and metadata
- Model Registration service processes the artifacts and stores them in appropriate versioned location
- Returns new model version identifier

**Example API Call**:
```python
register_model(
    model_path="/tmp/training_output/model_20250421_154500/",
    version_bump="patch",
    training_info={
        "dataset": {
            "name": "NW-test-camera-2",
            "sample_count": 240
        },
        "hyperparameters": {
            "learning_rate": 2.0e-5,
            "batch_size": 4,
            "epochs": 3
        }
    },
    performance_metrics={
        "validation": {
            "loss": 0.0532
        }
    }
)
```

### 4. Orchestration → All Components

**Purpose**: Coordinate system-wide events and processes

**Interface Type**: Event-driven API

**Contract**:
- Orchestration manages resources and schedules jobs
- Components register for events they need to respond to
- Components report status changes to Orchestration
- Orchestration handles error conditions and retries

**Event Types**:
- `dataset.new`: New dataset has been uploaded
- `dataset.validated`: Dataset validation complete
- `training.started`: Fine-tuning job has started
- `training.completed`: Fine-tuning job has completed
- `model.registered`: New model has been registered
- `model.promoted`: Model has been promoted to latest/stable

## Resource Allocation

### Compute Resources

The Orchestration service manages compute resource allocation for fine-tuning jobs:

1. **Resource Pools**:
   - GPU Pool: Available NVIDIA GPUs for training
   - CPU Pool: Resources for data preprocessing and validation

2. **Allocation Strategy**:
   - Priority-based job scheduling
   - Resource reservation for critical jobs
   - Automatic scaling based on job requirements

3. **Interface**:
   ```json
   {
     "job_id": "ft-20250421-15490",
     "resource_request": {
       "gpu_count": 1, 
       "gpu_type": "NVIDIA A100",
       "cpu_cores": 8,
       "memory_gb": 64
     },
     "estimated_duration_hours": 3,
     "priority": "normal"
   }
   ```

## Data Flow

The complete data flow through the system follows this sequence:

1. New data is uploaded to `/data/raw/{dataset_name}/`
2. Data Ingestion service detects new data and triggers validation
3. Data Validation service verifies format, quality, and structure
4. Validated data is stored in `/data/processed/{dataset_name}/`
5. Orchestration service schedules fine-tuning based on available resources
6. Fine-Tuning service trains the model using validated data
7. Model Registration registers the new model with versioning
8. Model evaluation determines if the new model should be promoted
9. If approved, the new model becomes the latest production version

## Component Responsibilities

### Data Ingestion Service
- Monitor for new data uploads
- Perform initial format verification
- Trigger validation workflow
- Log all incoming data

### Data Validation Service
- Validate structural correctness of data
- Verify domain-specific requirements
- Generate validation reports
- Prepare data for fine-tuning

### Fine-Tuning Service
- Execute training jobs with specific configurations
- Monitor training progress
- Save model checkpoints
- Log training metrics
- Output trained model artifacts

### Model Registration Service
- Version models using semantic versioning
- Store model artifacts with metadata
- Update model lineage information
- Generate model cards
- Manage symbolic links for latest/stable versions

### Orchestration Service
- Coordinate overall system workflow
- Manage compute resources
- Schedule jobs based on priority
- Handle error conditions and retries
- Monitor system health

## Configuration

The system is configured through structured YAML files in the `/config` directory:

```yaml
# Example orchestration configuration
orchestration:
  event_timeout: 30m
  retry_limit: 3
  scheduling:
    max_concurrent_jobs: 2
    priority_levels:
      - critical
      - high
      - normal
      - low
  monitoring:
    health_check_interval: 5m
    metrics_retention_days: 30
```

## Error Handling

Each component implements standardized error handling:

1. **Error Classifications**:
   - Transient: Temporary issues that can be retried
   - Permanent: Issues that require manual intervention
   - Resource: Issues related to resource availability

2. **Recovery Strategies**:
   - Automatic retry with exponential backoff for transient errors
   - Circuit breaker patterns to prevent cascading failures
   - Fallback mechanisms for critical operations

3. **Logging and Notification**:
   - Structured error logs with context information
   - Error events for system-wide monitoring
   - Alerts for critical failures

## Security Considerations

1. **Data Isolation**:
   - Raw data is isolated from processing components
   - Read-only access to processed data for training

2. **Model Protection**:
   - Base models are immutable
   - Versioned models have write-once semantics
   - Changes to "latest" require verification

3. **Access Control**:
   - Component-specific service accounts
   - Least-privilege principle
   - Audit logging for all operations 