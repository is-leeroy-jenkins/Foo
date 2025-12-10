BEGIN TRANSACTION;
CREATE TABLE IF NOT EXISTS "ChatLog" 
(
	"ChatLogId"	INTEGER NOT NULL UNIQUE,
	"Type"	TEXT(80),
	"Date"	TEXT(80),
	"Content"	TEXT(255),
	PRIMARY KEY("ChatLogId" AUTOINCREMENT)
);

CREATE TABLE IF NOT EXISTS "AgencyAccounts" 
(
	"AgencyAccountsId"	INTEGER NOT NULL UNIQUE,
	"TreasuryAgencyCode"	TEXT(80),
	"AgencyCode"	TEXT(80),
	"AgencyName"	TEXT(80),
	"BureauCode"	TEXT(80),
	"BureauName"	TEXT(80),
	"AccountCode"	TEXT(80),
	"AccountName"	TEXT(80),
	PRIMARY KEY("AgencyAccountsId" AUTOINCREMENT)
);

CREATE TABLE IF NOT EXISTS "Files" 
(
	"FilesId"	INTEGER NOT NULL UNIQUE,
	"filename"	TEXT(80),
	"id"	TEXT(80),
	"name"	TEXT(80),
	"extension"	TEXT(80),
	PRIMARY KEY("FilesId" AUTOINCREMENT)
);

CREATE TABLE IF NOT EXISTS "SF133" 
(
	"ID"	INTEGER NOT NULL UNIQUE,
	"FiscalYear"	TEXT(80),
	"BudgetAgencyCode"	TEXT(80),
	"BudgetAgencyName"	TEXT(80),
	"BudgetBureauCode"	TEXT(80),
	"BudgetBureauName"	TEXT(80),
	"MainAccountCode"	TEXT(80),
	"MainAccountName"	TEXT(80),
	"BudgetAccountCode"	TEXT(80),
	"TreasuryAccount"	TEXT(80),
	"TreasuryAppropriationFundSymbol"	TEXT(80),
	"TreasuryAgencyCode"	TEXT(80),
	"AllocationAccount"	TEXT(80),
	"TreasuryAccountCode"	TEXT(80),
	"BeginningPeriodOfAvailability"	TEXT(80),
	"EndingPeriodOfAvailability"	TEXT(80),
	"STAT"	TEXT(80),
	"CreditIndicator"	TEXT(80),
	"Cohort"	TEXT(80),
	"LineNumber"	TEXT(80),
	"LineDescription"	TEXT(80),
	"Category"	TEXT(80),
	"AgencyName"	TEXT(80),
	"SectionName"	TEXT(80),
	"SectionNumber"	TEXT(80),
	"LineType"	TEXT(80),
	"BudgetAccountTitle"	TEXT(80),
	"FinancingAccount"	TEXT(80),
	"November"	NUMERIC DEFAULT 0.0,
	"January"	NUMERIC DEFAULT 0.0,
	"Feburary"	NUMERIC DEFAULT 0.0,
	"April"	NUMERIC DEFAULT 0.0,
	"May"	NUMERIC DEFAULT 0.0,
	"July"	NUMERIC DEFAULT 0.0,
	"August"	NUMERIC DEFAULT 0.0,
	"Q1"	NUMERIC DEFAULT 0.0,
	"Q2"	NUMERIC DEFAULT 0.0,
	"Q3"	NUMERIC DEFAULT 0.0,
	"Q4"	NUMERIC DEFAULT 0.0,
	"LineName"	TEXT(80),
	"ProgramCategoryStub"	TEXT(80),
	"CategoryStub"	TEXT(80),
	PRIMARY KEY("ID" AUTOINCREMENT)
);

CREATE TABLE IF NOT EXISTS "OMB Circular A-11 Preparation Submission And Execution Of The Budget" 
(
	"Index"	INTEGER NOT NULL UNIQUE,
	"Question"	TEXT(80),
	"Answer"	TEXT(80),
	PRIMARY KEY("Index" AUTOINCREMENT)
);

CREATE TABLE IF NOT EXISTS "OMB Circular A-11 Section 120 Apportionment Process" 
(
	"Index"	INTEGER NOT NULL UNIQUE,
	"Question"	TEXT(80),
	"Answer"	TEXT(80),
	PRIMARY KEY("Index" AUTOINCREMENT)
);

CREATE TABLE IF NOT EXISTS "Apportionments" 
(
	"ApportionmentsId"	INTEGER NOT NULL UNIQUE,
	"FiscalYear"	TEXT(80),
	"BPOA"	TEXT(80),
	"EPOA"	TEXT(80),
	"MainAccount"	TEXT(80),
	"TreasuryAccountCode"	TEXT(80),
	"TreasuryAccountName"	TEXT(80),
	"AvailabilityType"	TEXT(80),
	"BudgetAccountCode"	TEXT(80),
	"BudgetAccountName"	TEXT(80),
	"LineNumber"	TEXT(80),
	"LineSplit"	TEXT(80),
	"LineName"	TEXT(80),
	"Amount"	DOUBLE DEFAULT 0.0,
	PRIMARY KEY("ApportionmentsId" AUTOINCREMENT)
);

CREATE TABLE IF NOT EXISTS "Appropriations" 
(
	"AppropriationsId"	INTEGER NOT NULL UNIQUE,
	"FiscalYear"	TEXT(80),
	"PublicLaw"	TEXT(80),
	"AppropriationTitle"	TEXT(80),
	"EnactedDate"	TEXT(80),
	"ExplanatoryComments"	TEXT(80),
	"Authority"	DOUBLE DEFAULT 0.0,
	PRIMARY KEY("AppropriationsId" AUTOINCREMENT)
);

CREATE TABLE IF NOT EXISTS "Endpoints" 
(
	"EndpointsId"	INTEGER NOT NULL UNIQUE,
	"API"	TEXT(80),
	"Name"	TEXT(80),
	"Location"	TEXT(80),
	PRIMARY KEY("EndpointsId" AUTOINCREMENT)
);

CREATE TABLE IF NOT EXISTS "OMB Circular A-11 SF-132" 
(
	"Index"	INTEGER NOT NULL UNIQUE,
	"Question"	TEXT(150),
	"Answer"	TEXT(255),
	PRIMARY KEY("Index" AUTOINCREMENT)
);

CREATE TABLE IF NOT EXISTS "Partitions" 
(
	"PartitionsId"	INTEGER NOT NULL UNIQUE,
	"FiscalYear"	TEXT(80),
	"BPOA"	TEXT(80),
	"EPOA"	TEXT(80),
	"Type"	TEXT(80),
	"TreasuryAccountCode"	TEXT(80),
	"MainAccount"	TEXT(80),
	"BudgetAccountCode"	TEXT(80),
	"Amount"	DOUBLE DEFAULT 0.0,
	"LineNumber"	TEXT(80),
	"LineName"	TEXT(80),
	PRIMARY KEY("PartitionsId" AUTOINCREMENT)
);

CREATE TABLE IF NOT EXISTS "Principles Of Federal Appropriations Law" 
(
	"Index"	INTEGER NOT NULL UNIQUE,
	"Question"	TEXT(150),
	"Answer"	TEXT(255),
	PRIMARY KEY("Index" AUTOINCREMENT)
);

CREATE TABLE IF NOT EXISTS "Prompts" 
(
	"PromptsId"	INTEGER NOT NULL UNIQUE,
	"Name"	TEXT(80),
	"Text"	TEXT(255),
	"Version"	INTEGER,
	"ID"	TEXT(80),
	PRIMARY KEY("PromptsId" AUTOINCREMENT)
);

CREATE TABLE IF NOT EXISTS "Title 31 Code Of Federal Regulations" 
(
	"Index"	INTEGER NOT NULL UNIQUE,
	"Question"	TEXT(150),
	"Answer"	TEXT(255),
	PRIMARY KEY("Index" AUTOINCREMENT)
);
COMMIT;
