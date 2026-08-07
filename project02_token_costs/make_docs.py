"""Generate the synthetic household-document corpus for project 02.

Five documents a typical middle-income US household actually deals with:
a lease, an insurance denial letter, an explanation of benefits, an
employee benefits summary, and a utility bill. All content is synthetic
(no real people, policies, or account numbers) but sized and worded to
match the real thing, so token counts are representative.

Run:  python make_docs.py   -> writes docs/*.txt
"""

from pathlib import Path

LEASE_CLAUSE = (
    "Section {n}. The Tenant shall maintain the Premises in a clean and "
    "sanitary condition and shall not make any alterations, additions, or "
    "improvements to the Premises without the prior written consent of the "
    "Landlord. Any such alteration, addition, or improvement made without "
    "consent shall become the property of the Landlord and shall remain "
    "upon and be surrendered with the Premises upon termination of this "
    "Agreement. The Tenant shall be responsible for the cost of returning "
    "the Premises to its original condition, normal wear and tear excepted, "
    "and such costs may be deducted from the security deposit as permitted "
    "under applicable state law.\n\n"
)

DENIAL_PARA = (
    "After careful review of claim {n}, we have determined that the service "
    "billed under procedure code 97110 does not meet the plan definition of "
    "medical necessity as described in your Evidence of Coverage, because "
    "the submitted documentation does not establish that a lower-cost "
    "alternative treatment was attempted prior to the service date. You "
    "have the right to appeal this determination within 180 days of receipt "
    "of this notice by submitting a written request, together with any "
    "additional clinical documentation, to the address listed on the back "
    "of your member identification card.\n\n"
)

EOB_LINE = (
    "Service line {n}: Office visit, established patient. Amount billed "
    "$310.00; plan discount $142.55; amount covered $118.20; deductible "
    "applied $28.00; coinsurance $21.25; your responsibility $49.25. This "
    "is not a bill. Your provider may bill you separately for the amount "
    "shown as your responsibility.\n\n"
)

BENEFITS_PARA = (
    "Item {n}. Eligible employees may enroll in the high-deductible health "
    "plan with an annual deductible of $3,300 for individual coverage and "
    "$6,600 for family coverage. The company contributes $750 annually to a "
    "health savings account for enrolled employees, prorated by hire date. "
    "Preventive care services received from in-network providers are "
    "covered at one hundred percent and are not subject to the deductible. "
    "Out-of-network services are subject to separate deductibles and "
    "balance billing, and prior authorization is required for non-emergency "
    "hospital admissions, advanced imaging, and specialty medications.\n\n"
)

UTILITY_PARA = (
    "Billing detail {n}: Meter read on 07/28 (actual). Usage this period: "
    "1,142 kWh over 30 days. Base charge $14.50; energy charge at $0.1312 "
    "per kWh $149.83; fuel cost adjustment $9.41; environmental compliance "
    "cost recovery $3.86; municipal franchise fee $5.32; state and local "
    "taxes $11.04. Average daily temperature this period was 4 degrees "
    "warmer than the same period last year.\n\n"
)

# Repetition counts size each file like its real-world counterpart:
# leases run 20-40 pages, employer benefits handbooks 50-80 pages,
# EOB statements accumulate lines over a plan year.
DOCS = {
    "lease_agreement.txt": ("RESIDENTIAL LEASE AGREEMENT\n\n", LEASE_CLAUSE, 70),
    "insurance_denial.txt": ("NOTICE OF ADVERSE BENEFIT DETERMINATION\n\n", DENIAL_PARA, 7),
    "explanation_of_benefits.txt": ("EXPLANATION OF BENEFITS, PLAN YEAR TO DATE\n\n", EOB_LINE, 30),
    "benefits_summary.txt": ("EMPLOYEE BENEFITS HANDBOOK, PLAN YEAR 2026\n\n", BENEFITS_PARA, 180),
    "utility_bill.txt": ("ELECTRIC SERVICE STATEMENT\n\n", UTILITY_PARA, 5),
}


def main():
    out = Path(__file__).parent / "docs"
    out.mkdir(exist_ok=True)
    for name, (header, para, reps) in DOCS.items():
        body = header + "".join(para.format(n=i + 1) for i in range(reps))
        (out / name).write_text(body, encoding="utf-8")
        print(f"wrote {name}: {len(body):,} chars")


if __name__ == "__main__":
    main()
