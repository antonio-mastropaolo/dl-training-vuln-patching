library(effsize)


t5_base<-read.csv("../Results/CrystalBLEU/t5-base-no-pt.csv",header=TRUE)
vulrepair<-read.csv("../Results/CrystalBLEU/vulrepair-retrained.csv",header=TRUE)
ss_hard_prompt<-read.csv("../Results/CrystalBLEU/self-supervised-plus-hard-prompt-tuning.csv",header=TRUE)
ss_soft_prompt<-read.csv("../Results/CrystalBLEU/self-supervised-plus-soft-prompt-tuning.csv",header=TRUE)
ss_bf_hard_prompt<-read.csv("../Results/CrystalBLEU/self-supervised-plus-supervised-bf-plus-hard-prompt-tuning.csv",header=TRUE)
ss_bf_soft_prompt<-read.csv("../Results/CrystalBLEU/self-supervised-plus-supervised-bf-plus-soft-prompt-tuning.csv",header=TRUE)
ss_bf_vuln<-read.csv("../Results/CrystalBLEU/self-supervised-plus-supervised-bf-plus-supervised-vuln.csv",header=TRUE)



#############################################################
# res=list(Wilcoxon.p=c())
#res$Wilcoxon.p=c(wilcox.test(t5_base$beam_1,vulrepair$beam_1,alternative="two.side",paired=TRUE)$p.value)
#cliff.delta(t5_base$beam_1,vulrepair$beam_1)
#res=data.frame(res)
#print(res)
##############################################################

##############################################################
# res=list(Wilcoxon.p=c())
# res$Wilcoxon.p=c(wilcox.test(vulrepair$beam_1,ss_bf_vuln$beam_1,alternative="two.side",paired=TRUE)$p.value)
# cliff.delta(vulrepair$beam_1,ss_bf_vuln$beam_1)
# res=data.frame(res)
#print(res)
#############################################################


res=list(Wilcoxon.p=c(), Wilcoxon.estimate=c())

res$Wilcoxon.p=(wilcox.test(vulrepair$beam_1,ss_soft_prompt$soft_1_beam_1,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(vulrepair$beam_1,ss_soft_prompt$soft_2_beam_1,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(vulrepair$beam_1,ss_soft_prompt$soft_3_beam_1,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(vulrepair$beam_1,ss_soft_prompt$soft_4_beam_1,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(vulrepair$beam_1,ss_soft_prompt$soft_5_beam_1,alternative="two.side",paired=TRUE)$p.value)

res$Wilcoxon.estimate=append(res$Wilcoxon.estimate, cliff.delta(vulrepair$beam_1,ss_soft_prompt$soft_1_beam_1)$estimate)
res$Wilcoxon.estimate=append(res$Wilcoxon.estimate, cliff.delta(vulrepair$beam_1,ss_soft_prompt$soft_2_beam_1)$estimate)
res$Wilcoxon.estimate=append(res$Wilcoxon.estimate, cliff.delta(vulrepair$beam_1,ss_soft_prompt$soft_3_beam_1)$estimate)
res$Wilcoxon.estimate=append(res$Wilcoxon.estimate, cliff.delta(vulrepair$beam_1,ss_soft_prompt$soft_4_beam_1)$estimate)
res$Wilcoxon.estimate=append(res$Wilcoxon.estimate, cliff.delta(vulrepair$beam_1,ss_soft_prompt$soft_5_beam_1)$estimate)


res=data.frame(res)
res$Wilcoxon.p=p.adjust(res$Wilcoxon.p, method="holm")
print(res)

# #############################################################



# #############################################################
res=list(Wilcoxon.p=c(), Wilcoxon.estimate=c())

res$Wilcoxon.p=(wilcox.test(vulrepair$beam_1,ss_hard_prompt$hard_1_beam_1,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(vulrepair$beam_1,ss_hard_prompt$hard_2_beam_1,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(vulrepair$beam_1,ss_hard_prompt$hard_3_beam_1,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(vulrepair$beam_1,ss_hard_prompt$hard_4_beam_1,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(vulrepair$beam_1,ss_hard_prompt$hard_5_beam_1,alternative="two.side",paired=TRUE)$p.value)

res$Wilcoxon.estimate=append(res$Wilcoxon.estimate, cliff.delta(vulrepair$beam_1,ss_hard_prompt$hard_1_beam_1)$estimate)
res$Wilcoxon.estimate=append(res$Wilcoxon.estimate, cliff.delta(vulrepair$beam_1,ss_hard_prompt$hard_2_beam_1)$estimate)
res$Wilcoxon.estimate=append(res$Wilcoxon.estimate, cliff.delta(vulrepair$beam_1,ss_hard_prompt$hard_3_beam_1)$estimate)
res$Wilcoxon.estimate=append(res$Wilcoxon.estimate, cliff.delta(vulrepair$beam_1,ss_hard_prompt$hard_4_beam_1)$estimate)
res$Wilcoxon.estimate=append(res$Wilcoxon.estimate, cliff.delta(vulrepair$beam_1,ss_hard_prompt$hard_5_beam_1)$estimate)


res=data.frame(res)
res$Wilcoxon.p=p.adjust(res$Wilcoxon.p, method="holm")
print(res)

# #############################################################

# #############################################################
res=list(Wilcoxon.p=c(), Wilcoxon.estimate=c())

res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(ss_bf_soft_prompt$soft_1_beam_1,ss_bf_vuln$beam_1,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(ss_bf_soft_prompt$soft_2_beam_1,ss_bf_vuln$beam_1,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(ss_bf_soft_prompt$soft_3_beam_1,ss_bf_vuln$beam_1,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(ss_bf_soft_prompt$soft_4_beam_1,ss_bf_vuln$beam_1,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(ss_bf_soft_prompt$soft_5_beam_1,ss_bf_vuln$beam_1,alternative="two.side",paired=TRUE)$p.value)

res$Wilcoxon.estimate=append(res$Wilcoxon.estimate, cliff.delta(ss_bf_soft_prompt$soft_1_beam_1,ss_bf_vuln$beam_1)$estimate)
res$Wilcoxon.estimate=append(res$Wilcoxon.estimate, cliff.delta(ss_bf_soft_prompt$soft_2_beam_1,ss_bf_vuln$beam_1)$estimate)
res$Wilcoxon.estimate=append(res$Wilcoxon.estimate, cliff.delta(ss_bf_soft_prompt$soft_3_beam_1,ss_bf_vuln$beam_1)$estimate)
res$Wilcoxon.estimate=append(res$Wilcoxon.estimate, cliff.delta(ss_bf_soft_prompt$soft_4_beam_1,ss_bf_vuln$beam_1)$estimate)
res$Wilcoxon.estimate=append(res$Wilcoxon.estimate, cliff.delta(ss_bf_soft_prompt$soft_5_beam_1,ss_bf_vuln$beam_1)$estimate)


res=data.frame(res)
res$Wilcoxon.p=p.adjust(res$Wilcoxon.p, method="holm")
print(res)

#############################################################


res=list(Wilcoxon.p=c(), Wilcoxon.estimate=c())

res$Wilcoxon.p=(wilcox.test(ss_bf_vuln$beam_1,ss_bf_hard_prompt$hard_1_beam_1,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(ss_bf_hard_prompt$hard_2_beam_1,ss_bf_vuln$beam_1,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(ss_bf_hard_prompt$hard_3_beam_1,ss_bf_vuln$beam_1,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(ss_bf_hard_prompt$hard_4_beam_1,ss_bf_vuln$beam_1,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(ss_bf_hard_prompt$hard_5_beam_1,ss_bf_vuln$beam_1,alternative="two.side",paired=TRUE)$p.value)



res$Wilcoxon.estimate=cliff.delta(ss_bf_vuln$beam_1,ss_bf_hard_prompt$hard_1_beam_1)$estimate
res$Wilcoxon.estimate=append(res$Wilcoxon.estimate, cliff.delta(ss_bf_hard_prompt$hard_2_beam_1,ss_bf_vuln$beam_1)$estimate)
res$Wilcoxon.estimate=append(res$Wilcoxon.estimate, cliff.delta(ss_bf_hard_prompt$hard_3_beam_1,ss_bf_vuln$beam_1)$estimate)
res$Wilcoxon.estimate=append(res$Wilcoxon.estimate, cliff.delta(ss_bf_hard_prompt$hard_4_beam_1,ss_bf_vuln$beam_1)$estimate)
res$Wilcoxon.estimate=append(res$Wilcoxon.estimate, cliff.delta(ss_bf_hard_prompt$hard_5_beam_1,ss_bf_vuln$beam_1)$estimate)


res=data.frame(res)
res$Wilcoxon.p=p.adjust(res$Wilcoxon.p, method="holm")
print(res)

#############################################################
