# Background: the patching problem TriageBand addresses

Read this first. It explains the problem, why it matters, and every term the
rest of the repository uses. No machine-learning background is assumed.

## Vocabulary

| Term | Meaning |
|---|---|
| **Vulnerability** | A flaw in software that an attacker could use to do something they shouldn't — run code, read data, crash a service. |
| **CVE** | *Common Vulnerabilities and Exposures.* A public ID for one specific vulnerability, e.g. `CVE-2024-3400`. Everything else in this list is keyed by it. |
| **Exploit** | Code or a technique that actually uses a vulnerability. A vulnerability with no working exploit is a risk on paper; one with an exploit is a risk in practice. |
| **Exploited in the wild** | Attackers have actually used it against real victims — not just in a lab. |
| **NVD** | The US *National Vulnerability Database*. Publishes a record per CVE: description, severity, weakness type, affected products. |
| **CVSS** | *Common Vulnerability Scoring System.* A 0–10 severity rating: how bad a vulnerability *could* be if exploited. It says nothing about whether anyone *will* exploit it. |
| **CWE** | *Common Weakness Enumeration.* The type of mistake behind a vulnerability, e.g. `CWE-79` cross-site scripting, `CWE-787` out-of-bounds write. |
| **CPE** | *Common Platform Enumeration.* NVD's list of affected products and versions. Added during NVD's analysis, often days to months after the CVE is published. |
| **EPSS** | *Exploit Prediction Scoring System*, run by FIRST. A free daily probability that a CVE will be exploited in the next 30 days. The current industry standard for prioritisation. |
| **KEV** | CISA's *Known Exploited Vulnerabilities* catalogue — CVEs the US government has confirmed are exploited in the wild, each with the date it was added. Used here as the answer key. |
| **Proof of concept (PoC)** | A public demonstration exploit, often on GitHub. A strong early warning sign. |
| **Exploit-DB / Metasploit** | Public collections of exploits. Metasploit modules are ready-to-run attack tools. |
| **Patch budget / effort** | How many CVEs a team can remediate in a period, or that number as a share of all CVEs. |
| **Coverage** | Of all the CVEs that got exploited, the share a team had already patched. *Did we fix the ones that mattered?* |
| **Efficiency** | Of all the CVEs a team patched, the share that were actually exploited. *Was our effort well spent?* |
| **Point-in-time** | Using only information that existed at the moment a decision was made. A model tested with information from the future looks better than it will ever be in practice; that error is called **leakage**. |
| **Holdout** | Data set aside and never used to build or tune the model, so it gives an honest final test. |

## The problem

Software vulnerabilities are published faster than any organisation can fix
them. NVD published about **40,000 CVEs in 2024** alone (39,633 in this
project's data), and a single vulnerability scan of a mid-sized network
routinely reports thousands of open ones. Patching takes testing, change
windows and downtime, so a security team might remediate a few hundred a week.

The question is therefore never *"which vulnerabilities are bad?"* — most are,
on paper — but **"which ones do we fix first?"** That is triage.

Two facts make triage both hard and important:

1. **Very few vulnerabilities are ever exploited.** In this project's data, about
   **0.4%** of CVEs reach the KEV catalogue within a year of publication. A
   team that patches at random spends 99.6% of its effort on vulnerabilities no
   attacker uses.
2. **The ones that are exploited are exploited fast.** Mandiant found the average
   time from disclosure to exploitation fell to **5 days in 2023**, from 32 days
   the year before.[^mandiant] There is little time to get the order right.

## Why it matters

Exploiting vulnerabilities is a major way attackers get in. Verizon's 2024 Data
Breach Investigations Report found exploitation as the initial access path rose
**180%** year over year, reaching about **14%** of breaches — third after
stolen credentials and phishing — driven largely by mass exploitation such as
the MOVEit campaign.[^dbir]

The cost of bad triage falls on both sides:

- **Missed risk.** An exploited vulnerability left unpatched because it ranked
  low is how a breach starts.
- **Wasted effort.** Every patch spent on a vulnerability nobody exploits is
  capacity not spent on one that matters, and patching fatigue is real.

## How teams prioritise today

- **CVSS severity** is the traditional default — "fix everything 9.0 and above."
  It measures potential damage, not likelihood, and in this project's data it
  is close to useless at finding the vulnerabilities that actually get
  exploited.
- **EPSS** estimates likelihood directly, from signals such as whether exploit
  code exists, how the vulnerability is discussed, and its technical traits.
  It is far better than CVSS and is the bar any new method has to clear.

## What TriageBand does

TriageBand is a prioritiser built *on top of* EPSS rather than against it.
EPSS is very good at sorting vulnerabilities into broad risk bands and much
weaker at ordering them within a band. TriageBand keeps EPSS's bands and uses
its own model — trained only on information available when a CVE is
published — to decide the order inside each band. The result is a patch list
that catches more of the vulnerabilities that actually get exploited for the
same amount of work.

Results and methods are in [`README.md`](./README.md) and
[`FINDINGS.md`](./FINDINGS.md).

[^mandiant]: Mandiant / Google Cloud Threat Intelligence, *How Low Can You Go? An
    Analysis of 2023 Time-to-Exploit Trends* (October 2024). 138 vulnerabilities
    disclosed in 2023 and exploited in the wild.
    https://cloud.google.com/blog/topics/threat-intelligence/time-to-exploit-trends-2023
[^dbir]: Verizon, *2024 Data Breach Investigations Report*; covered in SecurityWeek,
    *Verizon DBIR 2024 Shows Surge in Vulnerability Exploitation*.
    https://www.securityweek.com/verizon-dbir-2024-shows-surge-in-vulnerability-exploitation-confirmed-data-breaches/
