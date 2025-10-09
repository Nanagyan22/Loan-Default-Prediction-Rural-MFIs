import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import logging

# Get database URL from environment
DATABASE_URL = os.getenv('DATABASE_URL')

Base = declarative_base()

# Lazy initialization of engine and session
engine = None
SessionLocal = None

def get_engine():
    """Get or create database engine"""
    global engine
    if engine is None:
        if not DATABASE_URL:
            # Use SQLite fallback for demo/development
            import warnings
            warnings.warn("DATABASE_URL not set, using SQLite fallback")
            engine = create_engine('sqlite:///./credit_scoring.db')
        else:
            engine = create_engine(DATABASE_URL)
    return engine

def get_session_local():
    """Get or create session maker"""
    global SessionLocal
    if SessionLocal is None:
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return SessionLocal

class Prediction(Base):
    """Store individual borrower predictions"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, index=True)
    applicant_id = Column(String, index=True)
    prediction_date = Column(DateTime, default=datetime.utcnow)
    
    # Borrower Information
    age = Column(Integer)
    gender = Column(String)
    marital_status = Column(String)
    education_level = Column(String)
    region = Column(String, index=True)
    occupation = Column(String, index=True)
    monthly_income = Column(Float)
    loan_amount_requested = Column(Float)
    loan_term_months = Column(Integer)
    household_size = Column(Integer)
    
    # Prediction Results
    model_used = Column(String, index=True)
    default_probability = Column(Float, index=True)
    risk_band = Column(String, index=True)
    recommendation = Column(String)
    
    # Actual Outcome (to be updated later)
    actual_default = Column(Boolean, nullable=True)
    actual_outcome_date = Column(DateTime, nullable=True)
    
    # Additional metadata
    feature_importance = Column(JSON, nullable=True)
    created_by = Column(String, default='system')

class PerformanceMetric(Base):
    """Track model performance metrics over time"""
    __tablename__ = 'performance_metrics'
    
    id = Column(Integer, primary_key=True, index=True)
    metric_date = Column(DateTime, default=datetime.utcnow, index=True)
    model_name = Column(String, index=True)
    
    # Metrics
    total_predictions = Column(Integer)
    total_actual_defaults = Column(Integer, nullable=True)
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    auc_roc = Column(Float, nullable=True)
    
    # Fairness metrics
    gender_disparity = Column(Float, nullable=True)
    regional_disparity = Column(Float, nullable=True)
    occupational_disparity = Column(Float, nullable=True)
    
    # Additional notes
    notes = Column(Text, nullable=True)

class FairnessAlert(Base):
    """Track fairness and bias alerts"""
    __tablename__ = 'fairness_alerts'
    
    id = Column(Integer, primary_key=True, index=True)
    alert_date = Column(DateTime, default=datetime.utcnow, index=True)
    alert_type = Column(String, index=True)  # 'gender', 'region', 'occupation'
    severity = Column(String, index=True)  # 'low', 'medium', 'high', 'critical'
    
    # Alert details
    group_affected = Column(String)
    metric_value = Column(Float)
    threshold = Column(Float)
    description = Column(Text)
    
    # Resolution
    resolved = Column(Boolean, default=False)
    resolution_date = Column(DateTime, nullable=True)
    resolution_notes = Column(Text, nullable=True)

class BatchProcessing(Base):
    """Track batch processing jobs"""
    __tablename__ = 'batch_processing'
    
    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(String, unique=True, index=True)
    upload_date = Column(DateTime, default=datetime.utcnow)
    
    # Batch details
    total_records = Column(Integer)
    successful_predictions = Column(Integer)
    failed_predictions = Column(Integer)
    model_used = Column(String)
    
    # Summary statistics
    avg_default_probability = Column(Float)
    high_risk_count = Column(Integer)
    approval_rate = Column(Float)
    
    # File info
    original_filename = Column(String)
    processed_filename = Column(String)
    
    # Status
    status = Column(String, default='completed')  # 'pending', 'processing', 'completed', 'failed'
    error_message = Column(Text, nullable=True)

# Create all tables
def init_db():
    """Initialize database tables"""
    try:
        _engine = get_engine()
        Base.metadata.create_all(bind=_engine)
        logging.info("Database tables created successfully")
        return True
    except Exception as e:
        logging.error(f"Error creating database tables: {str(e)}")
        return False

# Database session management
def get_db():
    """Get database session"""
    _session_maker = get_session_local()
    db = _session_maker()
    try:
        yield db
    finally:
        db.close()

# Data access functions
def save_prediction(prediction_data: dict):
    """Save prediction to database"""
    _session_maker = get_session_local()
    db = _session_maker()
    try:
        prediction = Prediction(**prediction_data)
        db.add(prediction)
        db.commit()
        db.refresh(prediction)
        return prediction.id
    except Exception as e:
        db.rollback()
        logging.error(f"Error saving prediction: {str(e)}")
        raise
    finally:
        db.close()

def get_predictions(limit: int = 100, region: str = None, model: str = None):
    """Retrieve predictions with optional filtering"""
    _session_maker = get_session_local()
    db = _session_maker()
    try:
        query = db.query(Prediction)
        
        if region:
            query = query.filter(Prediction.region == region)
        if model:
            query = query.filter(Prediction.model_used == model)
        
        predictions = query.order_by(Prediction.prediction_date.desc()).limit(limit).all()
        return predictions
    except Exception as e:
        logging.error(f"Error retrieving predictions: {str(e)}")
        return []
    finally:
        db.close()

def get_fairness_stats(group_by: str = 'region'):
    """Get fairness statistics grouped by specified dimension"""
    _session_maker = get_session_local()
    db = _session_maker()
    try:
        from sqlalchemy import func
        
        if group_by == 'region':
            group_col = Prediction.region
        elif group_by == 'gender':
            group_col = Prediction.gender
        elif group_by == 'occupation':
            group_col = Prediction.occupation
        else:
            group_col = Prediction.region
        
        stats = db.query(
            group_col.label('group'),
            func.count(Prediction.id).label('total'),
            func.avg(Prediction.default_probability).label('avg_default_prob'),
            func.sum(func.cast(Prediction.default_probability < 0.3, Integer)).label('approvals')
        ).group_by(group_col).all()
        
        # Convert to dictionaries
        result = []
        for stat in stats:
            approval_rate = (stat.approvals / stat.total * 100) if stat.total > 0 else 0
            result.append({
                'group': stat.group,
                'total': stat.total,
                'avg_default_prob': float(stat.avg_default_prob) if stat.avg_default_prob else 0,
                'approval_rate': approval_rate
            })
        
        return result
    except Exception as e:
        logging.error(f"Error getting fairness stats: {str(e)}")
        return []
    finally:
        db.close()

def save_batch_processing(batch_data: dict):
    """Save batch processing record"""
    _session_maker = get_session_local()
    db = _session_maker()
    try:
        batch = BatchProcessing(**batch_data)
        db.add(batch)
        db.commit()
        db.refresh(batch)
        return batch.batch_id
    except Exception as e:
        db.rollback()
        logging.error(f"Error saving batch processing: {str(e)}")
        raise
    finally:
        db.close()

def create_fairness_alert(alert_data: dict):
    """Create fairness alert"""
    _session_maker = get_session_local()
    db = _session_maker()
    try:
        alert = FairnessAlert(**alert_data)
        db.add(alert)
        db.commit()
        db.refresh(alert)
        return alert.id
    except Exception as e:
        db.rollback()
        logging.error(f"Error creating fairness alert: {str(e)}")
        raise
    finally:
        db.close()

def get_active_alerts():
    """Get active (unresolved) fairness alerts"""
    _session_maker = get_session_local()
    db = _session_maker()
    try:
        alerts = db.query(FairnessAlert).filter(
            FairnessAlert.resolved == False
        ).order_by(FairnessAlert.alert_date.desc()).all()
        return alerts
    except Exception as e:
        logging.error(f"Error retrieving alerts: {str(e)}")
        return []
    finally:
        db.close()

def update_actual_outcome(applicant_id: str, actual_default: bool):
    """Update actual default outcome for a prediction"""
    _session_maker = get_session_local()
    db = _session_maker()
    try:
        prediction = db.query(Prediction).filter(
            Prediction.applicant_id == applicant_id
        ).order_by(Prediction.prediction_date.desc()).first()
        
        if prediction:
            prediction.actual_default = actual_default
            prediction.actual_outcome_date = datetime.utcnow()
            db.commit()
            return True
        return False
    except Exception as e:
        db.rollback()
        logging.error(f"Error updating actual outcome: {str(e)}")
        return False
    finally:
        db.close()
