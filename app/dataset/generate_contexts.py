from pathlib import Path

DEPARTMENT_CONTEXTS = {
    "hr_context.txt": [
        "The HR department oversees employee onboarding, guiding new team members through orientation schedules, mentorship assignments, and completion of mandatory compliance training.",
        "Leave policies cover annual, sick, parental, and compassionate leave, with a streamlined digital approval workflow and clear escalation paths for urgent requests.",
        "Performance reviews combine quarterly check-ins, peer feedback, and goal-tracking dashboards to promote continuous development and transparent recognition programs.",
        "Recruitment initiatives emphasize inclusive hiring practices, structured interviews, and data-informed workforce planning aligned with long-term benefits administration strategies.",
    ],
    "finance_context.txt": [
        "The finance department manages annual budget allocation cycles, coordinating with department heads to prioritize capital expenditure and recurring operational costs.",
        "Expense reporting relies on card reconciliation, receipt digitization, and automated policy checks to reduce delays in procurement and invoice processing.",
        "Payment approvals integrate multi-level authorization, compliance audits, and tax documentation to maintain accountability across domestic and international transactions.",
        "Quarterly financial audits assess internal controls, vendor performance, and regulatory adherence while informing forward-looking tax compliance initiatives.",
    ],
    "engineering_context.txt": [
        "Engineering teams maintain CI/CD pipelines that automate unit testing, security scanning, and staged deployments for web and microservice architectures.",
        "Code review guidelines prioritize readability, performance benchmarks, and adherence to architectural patterns documented within the engineering handbook.",
        "Sprint planning balances feature delivery with technical debt remediation, using burn-down analytics and cross-functional backlog grooming sessions.",
        "Deployment procedures include blue-green strategies, feature flag rollouts, and post-release monitoring tied to automated rollback criteria and testing requirements.",
    ],
    "it_support_context.txt": [
        "IT support operates a 24/7 ticket management queue, triaging incidents based on user impact and leveraging runbooks for rapid resolution of common issues.",
        "Password reset and account provisioning policies integrate identity verification, MFA enrollment, and compliance logging across SaaS platforms.",
        "Hardware requests involve asset tracking, imaging standards, and scheduled maintenance cycles for laptops, peripherals, and network equipment.",
        "Security incident response covers phishing containment, VPN configuration audits, and patch management aligned with system maintenance windows.",
    ],
}


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    for filename, paragraphs in DEPARTMENT_CONTEXTS.items():
        path = base_dir / filename
        path.write_text("\n\n".join(paragraphs), encoding="utf-8")
        print(f"✅ wrote {path.name} with {len(paragraphs)} paragraphs")
    print("🎉 department context files ready")


if __name__ == "__main__":
    main()
