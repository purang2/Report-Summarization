# Report-Summarization
Summarize Long Document with Pretrained sequence-to-sequence LM with long-range attention! 







## Trained Dataset 

### INPUT 


Google Docs 기준 8 pages 정도 분량을 입력으로 받는다.
책이나 A4지 기준으로 한 페이지에 500 words 정도 됩니다.



There are some similarities in how Medicare pays ASCs and hospital outpatient departments for the procedures they perform. However, the methods used by CMS to calculate the payment rates in each system, as well as the mechanisms used to revise the Medicare payment rates, differ. In 1980, legislation was enacted that enabled ASCs to bill Medicare for certain surgical procedures provided to Medicare beneficiaries. Under the ASC payment system, Medicare pays a predetermined, and generally all- inclusive, amount per procedure to the facility. The approximately 2,500 surgical procedures that ASCs may bill for under Medicare are assigned to one of nine payment groups that contain procedures with similar costs, but not necessarily clinical similarities. All procedures assigned to one payment group are paid at the same rate. Under the Medicare payment system, when more than one procedure is performed at the same time, the ASC receives a payment for each of the procedures. However, the procedure that has the highest payment rate receives 100 percent of the applicable payment, and each additional procedure receives 50 percent of the applicable payment. The Medicare payment for a procedure performed at an ASC is intended to cover the direct costs for a procedure, such as nursing and technician services, drugs, medical and surgical supplies and equipment, anesthesia materials, and diagnostic services (including imaging services), and the indirect costs associated with the procedure, including use of the facility and related administrative services. The ASC payment for a procedure does not include payment for implantable devices or prosthetics related to the procedure; ASCs may bill separately for those items. In addition, the payment to the ASC does not include payment for professional services associated with the procedure; the physician who performs the procedure and the anesthesiologist or anesthetist bill Medicare directly for their services. Finally, the ASC payment does not include payment for certain other services that are not directly related to performing the procedure and do not occur during the time that the procedure takes place, such as some laboratory, X-ray, and other diagnostic tests. Because these additional services are not ASC procedures, they may be performed by another provider. In those cases, Medicare makes payments to those providers for the additional services. For example, a laboratory service needed to evaluate a tissue sample removed during an ASC procedure is not included in the ASC payment. The provider that evaluated the tissue sample would bill and receive payment from Medicare for that service. Because ASCs receive one inclusive payment for the procedure performed and its associated services, such as drugs, they generally include on their Medicare claim only the procedure performed. In 1997, legislation was enacted that required the implementation of a prospective payment system for hospital outpatient departments; the OPPS was implemented in August 2000. Although ASCs perform only procedures, hospital outpatient departments provide a much broader array of services, including diagnostic services, such as X-rays and laboratory tests, and emergency room and clinic visits. Each of the approximately 5,500 services, including procedures, that hospital outpatient departments perform is assigned to one of over 800 APC groups with other services with clinical and cost similarities for payment under the OPPS. All services assigned to one APC group are paid the same rate. Similar to ASCs, when hospitals perform multiple procedures at the same time, they receive 100 percent of the applicable payment for the procedure that has the highest payment rate, and 50 percent of the applicable payment for each additional procedure, subject to certain exceptions. Like payments to ASCs, payment for a procedure under the OPPS is intended to cover the costs of the use of the facility, nursing and technician services, most drugs, medical and surgical supplies and equipment, anesthesia materials, and administrative costs. Medicare payment to a hospital for a procedure does not include professional services for physicians or other nonphysician practitioners. These services are paid for separately by Medicare. However, there are some differences between ASC and OPPS payments for procedures. Under the OPPS, hospital outpatient departments generally may not bill separately for implantable devices related to the procedure, but they may bill separately for additional services that are directly related to the procedure, such as certain drugs and diagnostic services, including X-rays. Hospital outpatient departments also may bill separately for additional services that are not directly related to the procedure and do not occur during the procedure, such as laboratory services to evaluate a tissue sample. Because they provide a broader array of services, and because CMS has encouraged hospitals to report all services provided during a procedure on their Medicare claims for rate-setting purposes, hospital claims may provide more detail about the services delivered during a procedure than ASC claims do. CMS set the initial 1982 ASC payment rates based on cost and charge data from 40 ASCs. At that time, there were about 125 ASCs in operation. Procedures were placed into four payment groups, and all procedures in a group were paid the same rate. When the ASC payment system was first established, federal law required CMS to review the payment rates periodically. In 1986, CMS conducted an ASC survey to gather cost and charge data. In 1990, using these data, CMS revised the payment rates and increased the number of payment groups to eight. A ninth payment group was established in 1991. These groups are still in use, although some procedures have been added to or deleted from the ASC-approved list. Although payments have not been revised using ASC cost data since 1990, the payment rates have been periodically updated for inflation. In 1994, Congress required that CMS conduct a survey of ASC costs no later than January 1, 1995, and thereafter every 5 years, to revise ASC payment rates. CMS conducted a survey in 1994 to collect ASC cost data. In 1998, CMS proposed revising ASC payment rates based on the 1994 survey data and assigned procedures performed at ASCs into payment groups that were comparable to the payment groups it was developing for the same procedures under the OPPS. However, CMS did not implement the proposal, and, as a result, the ASC payment system was not revised using the 1994 data. In 2003, MMA eliminated the requirement to conduct ASC surveys every 5 years and required CMS to implement a revised ASC payment system no later than January 1, 2008. During the course of our work, in August 2006, CMS published a proposed rule that would revise the ASC payment system effective January 1, 2008. In this proposed rule, CMS bases the revised ASC payment rates on the OPPS APC groups. However, the payment rates would be lower for ASCs. The initial OPPS payment rates, implemented in August 2000, were based on hospitals’ 1996 costs. To determine the OPPS payment rates, CMS first calculates each hospital’s cost for each service by multiplying the charge for that service by a cost-to-charge ratio computed from the hospital’s most recently reported data. After calculating the cost of each service for each hospital, the services are grouped by their APC assignment, and a median cost for each APC group is calculated from the median costs of all services assigned to it. Using the median cost, CMS assigns each APC group a weight based on its median cost relative to the median cost of all other APCs. To obtain a payment rate for each APC group, CMS multiplies the relative weight by a factor that converts it to a dollar amount. Beginning in 2002, as required by law, the APC group payment rates have been revised annually based on the latest charge and cost data. ... [(See more)]()



### OUTPUT 

Medicare pays for surgical procedures performed at ambulatory surgical centers (ASC) and hospital outpatient departments through different payment systems. Although they perform a similar set of procedures, no comparison of ASC and hospital outpatient per-procedure costs has been conducted. The Medicare Prescription Drug, Improvement, and Modernization Act of 2003 directed GAO to compare the relative costs of procedures furnished in ASCs to the relative costs of those procedures furnished in hospital outpatient departments, in particular, how accurately the payment groups used in the hospital outpatient prospective payment system (OPPS) reflect the relative costs of procedures performed in ASCs. To do this, GAO collected data from ASCs through a survey. GAO also obtained hospital outpatient data from the Centers for Medicare & Medicaid Services (CMS). GAO determined that the payment groups in the OPPS, known as ambulatory payment classification (APC) groups, accurately reflect the relative cost of procedures performed in ASCs. GAO calculated the ratio between each procedure's ASC median cost, as determined by GAO's survey, and the median cost of each procedure's corresponding APC group under the OPPS, referred to as the ASC-to-APC cost ratio. GAO also compared the OPPS median costs of those same procedures with the median costs of their APC groups, referred to as the OPPS-to-APC cost ratio. GAO's analysis of the ASC-to-APC and OPPS-to-APC cost ratios showed that 45 percent of all procedures in the analysis fell within a 0.10 point range of the ASC-to-APC median cost ratio, and 33 percent of procedures fell within a 0.10 point range of the OPPS-to-APC median cost ratio. These similar patterns of distribution around the median show that the APC groups reflect the relative costs of procedures provided by ASCs as well as they reflect the relative costs of procedures provided in hospital outpatient departments and can be used as the basis for the ASC payment system. GAO's analysis also identified differences in the cost of procedures in the two settings. The median cost ratio among all ASC procedures was 0.39 and when weighted by Medicare claims volume was 0.84. The median cost ratio for OPPS procedures was 1.04. Thus, the cost of procedures in ASCs is substantially lower than the corresponding cost in hospital outpatient departments.


