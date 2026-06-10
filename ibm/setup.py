from qiskit_ibm_runtime import QiskitRuntimeService
from dotenv import load_dotenv
import os

load_dotenv()

ibm_token = os.getenv("IBM_QUANTUM_TOKEN")
ibm_instance = os.getenv("IBM_QUANTUM_INSTANCE")

QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token=ibm_token,
    instance=ibm_instance,
    set_as_default=True,
    overwrite=True,
)

service = QiskitRuntimeService()  # loads your saved default account
backends = service.backends()
print([b.name for b in backends][:20])

# backend = service.backend("ibm_brisbane")   # example name; use one you see in service.backends()
# print(backend)
