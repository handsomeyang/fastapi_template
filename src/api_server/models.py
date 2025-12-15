from enum import Enum
from pydantic import BaseModel


class Job(str, Enum):
    admin = ("admin.",)
    unknown = ("unknown",)
    unemployed = ("unemployed",)
    management = ("management",)
    housemaid = ("housemaid",)
    entrepreneur = ("entrepreneur",)
    student = ("student",)
    blue_collar = ("blue-collar",)
    self_employed = ("self-employed",)
    retired = ("retired",)
    technician = ("technician",)
    services = "services"


class Marital(str, Enum):
    married = ("married",)
    divorced = ("divorced",)
    single = "single"


class Education(str, Enum):
    unknown = ("unknown",)
    secondary = ("secondary",)
    primary = ("primary",)
    tertiary = "tertiary"


class Yesno(str, Enum):
    yes = ("yes",)
    no = "no"


class Contact(str, Enum):
    unknown = ("unknown",)
    telephone = ("telephone",)
    cellular = "cellular"


class Month(str, Enum):
    jan = ("jan",)
    feb = ("feb",)
    mar = "mar"
    apr = ("apr",)
    may = ("may",)
    jun = ("jun",)
    jul = ("jul",)
    aug = ("aug",)
    sep = ("sep",)
    oct = ("oct",)
    nov = ("nov",)
    dec = "dec"


class Poutcome(str, Enum):
    unknown = ("unknown",)
    other = ("other",)
    failure = ("failure",)
    success = "success"


class CustomerData(BaseModel):
    age: int
    job: Job
    marital: Marital
    education: Education
    default: Yesno
    balance: float
    housing: Yesno
    loan: Yesno
    contact: Contact
    day: int
    month: Month
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: Poutcome


class HealthCheckResult(BaseModel):
    status: str


class PredictionResult(BaseModel):
    status: str
    prediction: str
