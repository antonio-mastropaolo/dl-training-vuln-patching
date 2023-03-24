library(exact2x2)
library(effsize)
library(xtable)

#################################################################################################
res=list(Dataset=c(),McNemar.p=c(),McNemar.OR=c())
d<-"mcnemar_file.csv"
t<-read.csv(d)

# m=mcnemar.exact(t$t5_base,t$vulrepair)
# res$Dataset=c(res$Dataset,as.character(d))
# res$McNemar.p=c(res$McNemar.p, m$p.value)
# res$McNemar.OR=c(res$McNemar.OR,m$estimate)

# res=data.frame(res)
# #p-value adjustment
# res$McNemar.p=p.adjust(res$McNemar.p,method="holm")
# print(res)
#################################################################################################

# res=list(Dataset=c(),McNemar.p=c(),McNemar.OR=c())

# m=mcnemar.exact(t$vulrepair,t$ss_and_bf)
# res$Dataset=c(res$Dataset,as.character(d))
# res$McNemar.p=c(res$McNemar.p, m$p.value)
# res$McNemar.OR=c(res$McNemar.OR,m$estimate)

# res=data.frame(res)
# #p-value adjustment
# #res$McNemar.p=p.adjust(res$McNemar.p,method="holm")
# print(res)
#################################################################################################

res=list(Dataset=c(),McNemar.p=c(),McNemar.OR=c())

m=mcnemar.exact(t$vulrepair,t$soft_prompt_tuning1)
res$Dataset=c(res$Dataset,as.character(d))
res$McNemar.p=c(res$McNemar.p, m$p.value)
res$McNemar.OR=c(res$McNemar.OR,m$estimate)

m=mcnemar.exact(t$vulrepair,t$soft_prompt_tuning2)
res$Dataset=c(res$Dataset,as.character(d))
res$McNemar.p=c(res$McNemar.p, m$p.value)
res$McNemar.OR=c(res$McNemar.OR,m$estimate)

m=mcnemar.exact(t$vulrepair,t$soft_prompt_tuning3)
res$Dataset=c(res$Dataset,as.character(d))
res$McNemar.p=c(res$McNemar.p, m$p.value)
res$McNemar.OR=c(res$McNemar.OR,m$estimate)

m=mcnemar.exact(t$vulrepair,t$soft_prompt_tuning4)
res$Dataset=c(res$Dataset,as.character(d))
res$McNemar.p=c(res$McNemar.p, m$p.value)
res$McNemar.OR=c(res$McNemar.OR,m$estimate)

m=mcnemar.exact(t$vulrepair,t$soft_prompt_tuning5)
res$Dataset=c(res$Dataset,as.character(d))
res$McNemar.p=c(res$McNemar.p, m$p.value)
res$McNemar.OR=c(res$McNemar.OR,m$estimate)

res=data.frame(res)
#p-value adjustment
res$McNemar.p=p.adjust(res$McNemar.p,method="holm")
print(res)

#################################################################################################

res=list(Dataset=c(),McNemar.p=c(),McNemar.OR=c())

m=mcnemar.exact(t$vulrepair,t$hard_prompt_tuning1)
res$Dataset=c(res$Dataset,as.character(d))
res$McNemar.p=c(res$McNemar.p, m$p.value)
res$McNemar.OR=c(res$McNemar.OR,m$estimate)

m=mcnemar.exact(t$vulrepair,t$hard_prompt_tuning2)
res$Dataset=c(res$Dataset,as.character(d))
res$McNemar.p=c(res$McNemar.p, m$p.value)
res$McNemar.OR=c(res$McNemar.OR,m$estimate)

m=mcnemar.exact(t$vulrepair,t$hard_prompt_tuning3)
res$Dataset=c(res$Dataset,as.character(d))
res$McNemar.p=c(res$McNemar.p, m$p.value)
res$McNemar.OR=c(res$McNemar.OR,m$estimate)

m=mcnemar.exact(t$vulrepair,t$hard_prompt_tuning4)
res$Dataset=c(res$Dataset,as.character(d))
res$McNemar.p=c(res$McNemar.p, m$p.value)
res$McNemar.OR=c(res$McNemar.OR,m$estimate)

m=mcnemar.exact(t$vulrepair,t$hard_prompt_tuning5)
res$Dataset=c(res$Dataset,as.character(d))
res$McNemar.p=c(res$McNemar.p, m$p.value)
res$McNemar.OR=c(res$McNemar.OR,m$estimate)

res=data.frame(res)
#p-value adjustment
res$McNemar.p=p.adjust(res$McNemar.p,method="holm")
print(res)

#################################################################################################

res=list(Dataset=c(),McNemar.p=c(),McNemar.OR=c())

m=mcnemar.exact(t$ss_bf_soft_prompt_tuning1,t$ss_and_bf)
res$Dataset=c(res$Dataset,as.character(d))
res$McNemar.p=c(res$McNemar.p, m$p.value)
res$McNemar.OR=c(res$McNemar.OR,m$estimate)

m=mcnemar.exact(t$ss_bf_soft_prompt_tuning2,t$ss_and_bf)
res$Dataset=c(res$Dataset,as.character(d))
res$McNemar.p=c(res$McNemar.p, m$p.value)
res$McNemar.OR=c(res$McNemar.OR,m$estimate)

m=mcnemar.exact(t$ss_bf_soft_prompt_tuning3,t$ss_and_bf)
res$Dataset=c(res$Dataset,as.character(d))
res$McNemar.p=c(res$McNemar.p, m$p.value)
res$McNemar.OR=c(res$McNemar.OR,m$estimate)

m=mcnemar.exact(t$ss_bf_soft_prompt_tuning4,t$ss_and_bf)
res$Dataset=c(res$Dataset,as.character(d))
res$McNemar.p=c(res$McNemar.p, m$p.value)
res$McNemar.OR=c(res$McNemar.OR,m$estimate)

m=mcnemar.exact(t$ss_bf_soft_prompt_tuning5,t$ss_and_bf)
res$Dataset=c(res$Dataset,as.character(d))
res$McNemar.p=c(res$McNemar.p, m$p.value)
res$McNemar.OR=c(res$McNemar.OR,m$estimate)


res=data.frame(res)
#p-value adjustment
res$McNemar.p=p.adjust(res$McNemar.p,method="holm")
print(res)

#################################################################################################

res=list(Dataset=c(),McNemar.p=c(),McNemar.OR=c())

m=mcnemar.exact(t$ss_bf_hard_prompt_tuning1,t$ss_and_bf)
res$Dataset=c(res$Dataset,as.character(d))
res$McNemar.p=c(res$McNemar.p, m$p.value)
res$McNemar.OR=c(res$McNemar.OR,m$estimate)

m=mcnemar.exact(t$ss_bf_hard_prompt_tuning2,t$ss_and_bf)
res$Dataset=c(res$Dataset,as.character(d))
res$McNemar.p=c(res$McNemar.p, m$p.value)
res$McNemar.OR=c(res$McNemar.OR,m$estimate)

m=mcnemar.exact(t$ss_bf_hard_prompt_tuning3,t$ss_and_bf)
res$Dataset=c(res$Dataset,as.character(d))
res$McNemar.p=c(res$McNemar.p, m$p.value)
res$McNemar.OR=c(res$McNemar.OR,m$estimate)

m=mcnemar.exact(t$ss_bf_hard_prompt_tuning4,t$ss_and_bf)
res$Dataset=c(res$Dataset,as.character(d))
res$McNemar.p=c(res$McNemar.p, m$p.value)
res$McNemar.OR=c(res$McNemar.OR,m$estimate)

m=mcnemar.exact(t$ss_bf_hard_prompt_tuning5,t$ss_and_bf)
res$Dataset=c(res$Dataset,as.character(d))
res$McNemar.p=c(res$McNemar.p, m$p.value)
res$McNemar.OR=c(res$McNemar.OR,m$estimate)

res=data.frame(res)
#p-value adjustment
res$McNemar.p=p.adjust(res$McNemar.p,method="holm")
print(res)

