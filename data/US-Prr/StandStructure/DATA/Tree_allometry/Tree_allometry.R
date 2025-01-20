# This script creats various plots/histograms from the original data sets
# Hideki Kobayashi, Oct 6, 2022

library(ggplot2)
library(reshape2)
library(gridExtra)
library(Rmisc)

###############################################################################
# Allometry sampling 2012 (Ex1)
almtry_sum = read.csv("allometry_summary.csv")
almtry_sum_odr =transform(almtry_sum, Tree.. = factor(Tree.., 
                                                      levels = c("pfrr_tree_1", "pfrr_tree_2", "pfrr_tree_3",
                                                                 "pfrr_tree_4", "pfrr_tree_5", "pfrr_tree_6",
                                                                 "pfrr_tree_7", "pfrr_tree_8", "pfrr_tree_9",
                                                                 "pfrr_tree_10", "pfrr_tree_11", "pfrr_tree_12",
                                                                 "pfrr_tree_13", "pfrr_tree_14", "pfrr_tree_15",
                                                                 "pfrr_tree_16", "pfrr_tree_17", "pfrr_tree_18",
                                                                 "pfrr_tree_19", "pfrr_tree_20", "pfrr_tree_21",
                                                                 "pfrr_tree_22")))
# only cut data
almtry_sum_cut = subset(almtry_sum_odr,almtry_sum_odr$cut.or.not.=="cut")

# total weight
a1 = ggplot(almtry_sum_cut,aes(x = Tree.., y = Total.Mass..DW.g.)) +
  geom_bar(stat = "identity") +
  scale_y_log10() +
  ylab("Toral weight (DW g)") +
  xlab("") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1),
        panel.background = element_rect(fill = "transparent", colour = "black"),
        text = element_text(size = 32))
ggsave(file = "Figures/allometry_summary_TotalDW.pdf", width = 10)
ggsave(file = "Figures/allometry_summary_TotalDW.jpg", width = 10)
show(a1)

# total Height
a2 = ggplot(almtry_sum_cut,aes(x = Tree.., y = H..m.)) +
  geom_bar(stat = "identity") +
  ylab("Tree Height (m)") +
  xlab("") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1),
        panel.background = element_rect(fill = "transparent", colour = "black"),
        text = element_text(size = 32))
ggsave(file = "Figures/allometry_summary_TH.pdf", width = 10)
ggsave(file = "Figures/allometry_summary_TH.jpg", width = 10)
show(a2)

# DBH
a3 = ggplot(almtry_sum_cut,aes(x = Tree.., y = DBH..cm.)) +
  geom_bar(stat = "identity") +
  ylab("DBH (cm)") +
  xlab("") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1),
        panel.background = element_rect(fill = "transparent", colour = "black"),
        text = element_text(size = 32))
ggsave(file = "Figures/allometry_summary_DBH.pdf", width = 10)
ggsave(file = "Figures/allometry_summary_DBH.jpg", width = 10)
show(a3)

# Total leaf area
a4 = ggplot(almtry_sum_cut,aes(x = Tree.., y = Total.leaf.area..m2.)) +
  geom_bar(stat = "identity") +
  ylab("Leaf area (m^2)") +
  xlab("") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1),
        panel.background = element_rect(fill = "transparent", colour = "black"),
        text = element_text(size = 32))
ggsave(file = "Figures/allometry_summary_LA.pdf", width = 10)
ggsave(file = "Figures/allometry_summary_LA.jpg", width = 10)
show(a4)

# mass partitions

temp1 = melt(almtry_sum_cut, id.vars ="Tree..") 
temp2 = subset(temp1,temp1$variable =="Stem.Mass..DW.g." 
               | temp1$variable =="Branches.Mass..DW.g."
               | temp1$variable =="Foliage.Mass..DW.g.")
temp2$value = as.numeric(temp2$value)

a5 = ggplot(temp2,aes(x = Tree.., y = value, fill=variable)) +
  geom_bar(stat = "identity", position = "fill") +
  scale_fill_brewer(palette='Set1') +
  ylab("Dry mass %") +
  xlab("") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1),
        panel.background = element_rect(fill = "transparent", colour = "black",size=1),
        text = element_text(size = 32),  legend.position = "none",)
ggsave(file = "Figures/allometry_summary_DW_pct.pdf", width = 10)
ggsave(file = "Figures/allometry_summary_DW_pct.jpg", width = 10)
show(a5)

# branch profile
bra = read.csv("branch_drymass_profile.csv")
brasub = subset(bra,!is.na(bra$X0.0.0.5m_dead))

# create parameter
# ID 1 
dl = c("dead","live","dead","live","dead","live","dead","live","dead","live",
       "dead","live","dead","live","dead","live","dead","live","dead","live",
       "dead","live","dead","live","no")
class = c(0.5,0.5,1.0,1.0,1.5,1.5,2.0,2.0,2.5,2.5,3.0,3.0,
          3.5,3.5,4.0,4.0,4.5,4.5,5.0,5.0,5.5,5.5,6.0,6.0,-9999)
b = list()

for (i in 1:16){
  t=brasub[i,]
  t2=t(t)
  t3=cbind(t2[2:25],dl[1:24],class[1:24])
  colnames(t3)=c("value","variable","class")
  t3= as.data.frame((t3))
  t3[,1]=as.numeric(t3[,1])
  t3[,3]=as.numeric(t3[,3])
  
  b[[i]]=ggplot(t3,aes(x = class, y = value, fill=variable)) +
    geom_bar(stat = "identity") +
    ylim(0,1000) +
    coord_flip() +
    theme(panel.background = element_rect(fill = "transparent", colour = "black",size=1),
          text = element_text(size = 8),  axis.text = element_text(size = 8,color="black"),
          legend.position = "none",
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank())  +
    ylab("Dry mass (DWg)") +
    xlab("Height (m) ") +
    ggtitle(brasub$Tree..[i]) 
  
}
pdf("Figures/branch_drymass_profile.pdf",width = 12)
multiplot(plotlist = b, cols = 8)
dev.off()
jpeg("Figures/branch_drymass_profile.jpg",width = 1000,height=600)
multiplot(plotlist = b, cols = 8)
dev.off()


# leaf mass profile
lm = read.csv("Leafmass_profile.csv")
lmsub = subset(lm,!is.na(lm$X0.0.0.5m))

# create parameter
# ID 1 
dl = c("live","live","live","live","live","live","live","live","live","live","live","live","live")
class = c(0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,-9999)
b = list()

for (i in 1:16){
  t=lmsub[i,]
  t2=t(t)
  t3=cbind(t2[2:13],dl[1:12],class[1:12])
  colnames(t3)=c("value","variable","class")
  t3= as.data.frame((t3))
  t3[,1]=as.numeric(t3[,1])
  t3[,3]=as.numeric(t3[,3])
  
  b[[i]]=ggplot(t3,aes(x = class, y = value)) +
    geom_bar(stat = "identity") +
    ylim(0,200) +
    coord_flip() +
    theme(panel.background = element_rect(fill = "transparent", colour = "black",size=1),
          text = element_text(size = 8),  axis.text = element_text(size = 8,color="black"),
          legend.position = "none",
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank()) +
    ylab("Leaf dry mass (DWg)") +
    xlab("Height (m) ") +
    ggtitle(lmsub$Tree..[i]) 
  
}
pdf("Figures/Leafmass_profile.pdf",width = 12)
multiplot(plotlist = b, cols = 8)
dev.off()
jpeg("Figures/Leafmass_profile.jpg",,width = 1000,height=600)
multiplot(plotlist = b, cols = 8)
dev.off()

# leaf area profile
lm = read.csv("Leafarea_profile.csv")
lmsub = subset(lm,!is.na(lm$X0.0.0.5m))

# create parameter
# ID 1 
dl = c("live","live","live","live","live","live","live","live","live","live","live","live","live")
class = c(0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,-9999)
b = list()

for (i in 1:16){
  t=lmsub[i,]
  t2=t(t)
  t3=cbind(t2[2:13],dl[1:12],class[1:12])
  colnames(t3)=c("value","variable","class")
  t3= as.data.frame((t3))
  t3[,1]=as.numeric(t3[,1])
  t3[,3]=as.numeric(t3[,3])
  
  b[[i]]=ggplot(t3,aes(x = class, y = value)) +
    geom_bar(stat = "identity") +
    ylim(0,0.6) +
    coord_flip() +
    theme(panel.background = element_rect(fill = "transparent", colour = "black",size=1),
          text = element_text(size = 8),  axis.text = element_text(size = 8,color="black"),
          legend.position = "none",
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank())  +
    ylab("Leaf area mass (m^2)") +
    xlab("Height (m) ") +
    ggtitle(lmsub$Tree..[i])
}
pdf("Figures/Leafarea_profile.pdf",width = 12)
multiplot(plotlist = b, cols = 8)
dev.off()
jpeg("Figures/Leafarea_profile.jpg",width = 1000,height=600)
multiplot(plotlist = b, cols = 8)
dev.off()

# stem mass profile
lm = read.csv("stem_drymass_profile.csv")
lmsub = subset(lm,!is.na(lm$X0.0.0.5m))

# create parameter
# ID 1 
dl = c("live","live","live","live","live","live","live","live","live","live","live","live","live")
class = c(0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,-9999)
b = list()

for (i in 1:16){
  t=lmsub[i,]
  t2=t(t)
  t3=cbind(t2[2:13],dl[1:12],class[1:12])
  colnames(t3)=c("value","variable","class")
  t3= as.data.frame((t3))
  t3[,1]=as.numeric(t3[,1])
  t3[,3]=as.numeric(t3[,3])
  
  tm = 1
  if(max(t3$value)>100){tm = 2}
  if(max(t3$value)>200){tm = 5}
  if(max(t3$value)>500){tm = 10}
  if(max(t3$value)>1000){tm = 15}
  if(max(t3$value)>1500){tm = 20}
  
  b[[i]]=ggplot(t3,aes(x = class, y = value/100)) +
    geom_bar(stat = "identity") +
    ylim(0,tm) +
    coord_flip() +
    theme(panel.background = element_rect(fill = "transparent", colour = "black",size=1),
          text = element_text(size = 8),  axis.text = element_text(size = 8,color="black"),
          legend.position = "none",
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank())  +
    ylab("Stem mass (x 100 DWg)") +
    xlab("Height (m) ") +
    ggtitle(lmsub$Tree..[i])
}
pdf("Figures/stem_drymass_profile.pdf",width = 12)
multiplot(plotlist = b, cols = 8)
dev.off()
jpeg("Figures/stem_drymass_profile.jpg",width = 1000,height=600)
multiplot(plotlist = b, cols = 8)
dev.off()

# stem diameter profile
sm = read.csv("stem_diameter_profile.csv")
smsub = subset(sm,!is.na(sm$X0.0.0.5m_top))

# create parameter
# ID 1 
dl = c("top","middle","bottom","top","middle","bottom","top","middle","bottom",
       "top","middle","bottom","top","middle","bottom","top","middle","bottom",
       "top","middle","bottom","top","middle","bottom","top","middle","bottom",
       "top","middle","bottom","top","middle","bottom","top","middle","bottom")
class = c(0.5,0.5,0.5,1.0,1.0,1.0,1.5,1.5,1.5,2.0,2.0,2.0,2.5,2.5,2.5,
          3.0,3.0,3.0,3.5,3.5,3.5,4.0,4.0,4.0,4.5,4.5,4.5,5.0,5.0,5.0,
          5.5,5.5,5.5,6.0,6.0,6.0)
b = list()

for (i in 1:16){
  t=smsub[i,]
  t2=t(t)
  t3=cbind(t2[2:37],dl[1:36],class[1:36])
  colnames(t3)=c("value","variable","class")
  t3= as.data.frame((t3))
  t3[,1]=as.numeric(t3[,1])
  t3[,3]=as.numeric(t3[,3])
  t4=subset(t3,t3$variable=="middle")
  
  tm = 2
  if(max(t4$value)>2){tm = 4}
  if(max(t4$value)>4){tm = 8}
  if(max(t4$value)>8){tm = 10}
  
  b[[i]]=ggplot(t4,aes(x = class, y = value)) +
    geom_bar(stat = "identity") +
    ylim(0,tm) +
    coord_flip() +
    theme(panel.background = element_rect(fill = "transparent", colour = "black",size=1),
          text = element_text(size = 8),  axis.text = element_text(size = 8,color="black"),
          legend.position = "none",
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank()) +
    ylab("Stem diameter (cm)") +
    xlab("Height (m) ") +
    ggtitle(smsub$Tree..[i]) 
  
}
pdf("Figures/stem_diameter_profile.pdf",width = 12)
multiplot(plotlist = b, cols = 8)
dev.off()
jpeg("Figures/stem_diameter_profile.jpg",,width = 1000,height=600)
multiplot(plotlist = b, cols = 8)
dev.off()

