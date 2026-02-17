# This script creats various plots/histograms from the original data sets
# Hideki Kobayashi, Oct 6, 2022

library(ggplot2)
library(reshape2)
library(gridExtra)
library(Rmisc)

# Create figures

# Direct sampling of black spruce 

# Tree census measurements (Ex 1)
census = read.csv("../Tree_census/TreeCensus.csv")

census$TreeHeight_2014..m. =as.numeric(census$TreeHeight_2014..m.)
# Tree position
p = ggplot(census,aes(y=Y..S.N...m.,x = X..E.W...m.,size=TreeHeight_2010..m.,color=TreeHeight_2010..m.))  +
  geom_point() + 
  xlim(0,30) + 
  ylim(0,30) +
  ylab("South - North (m)") +
  xlab("West - East (m)") + 
  scale_x_reverse() +
  theme(
  panel.background = element_rect(fill = "transparent", colour = "black"),
  text = element_text(size = 36))
ggsave(file = "Ex1_Tree_census_position.pdf", width = 13)
ggsave(file = "Ex1_Tree_census_position.jpg", width = 13)

# DBH and Height
p1=ggplot(census,aes(x = DBH_2010..cm.)) +
  geom_histogram(aes(y=..density..),color="black", fill="red") +
  geom_density(alpha=.2, fill="#FF6666") +
  xlim(0,8) +
  ylim(0,0.3) +
  ylab("Density") +
  xlab("DBH (cm)") +
  geom_vline(aes(xintercept=mean(DBH_2010..cm.,na.rm=TRUE)),
             color="blue", linetype="dashed", size=1) +
  theme(
    panel.background = element_rect(fill = "transparent", colour = "black"),
    text = element_text(size = 36))
ggsave(file = "Ex1_DBH_2010.pdf", width = 10)
ggsave(file = "Ex1_DBH_2010.jpg", width = 10)

p2=ggplot(census,aes(x = census$TreeHeight_2010..m.)) +
  geom_histogram(aes(y=..density..),color="black", fill="blue") +
  geom_density(alpha=.2, fill="#FF6666") +
  xlim(0,8) +
  ylim(0,0.8) +
  ylab("Density") +
  xlab("Height (m)") +
  geom_vline(aes(xintercept=mean(TreeHeight_2010..m.,na.rm=TRUE)),
             color="blue", linetype="dashed", size=1) +
 theme(
  panel.background = element_rect(fill = "transparent", colour = "black"),
  text = element_text(size = 36))
ggsave(file = "Ex1_TH_2010.pdf", width = 10)
ggsave(file = "Ex1_TH_2010.jpg", width = 10)

p3=ggplot(census,aes(x = DBH2014..cm.)) +
  geom_histogram(aes(y=..density..),color="black", fill="red") +
  geom_density(alpha=.2, fill="#FF6666") +
  xlim(0,8) +
  ylim(0,0.3) +
  ylab("Density") +
  xlab("DBH (cm)") +
  geom_vline(aes(xintercept=mean(DBH2014..cm.,na.rm=TRUE)),
             color="blue", linetype="dashed", size=1) +
theme(
  panel.background = element_rect(fill = "transparent", colour = "black"),
  text = element_text(size = 36))
ggsave(file = "Ex1_DBH_2014.pdf", width = 10)
ggsave(file = "Ex1_DBH_2014.jpg", width = 10)

p4=ggplot(census,aes(x = census$TreeHeight_2014..m.)) +
  geom_histogram(aes(y=..density..),color="black", fill="blue") +
  geom_density(alpha=.2, fill="#FF6666") +
  xlim(0,8) +
  ylim(0,0.8) +
  ylab("Density") +
  xlab("Height (m)") +
  geom_vline(aes(xintercept=mean(census$TreeHeight_2014..m.,na.rm=TRUE)),
             color="blue", linetype="dashed", size=1) +
  theme(
    panel.background = element_rect(fill = "transparent", colour = "black"),
    text = element_text(size = 36))
ggsave(file = "Ex1_TH_2014.pdf", width = 10)
ggsave(file = "Ex1_TH_2014.jpg", width = 10)

###############################################################################
# Allometry sampling 2012 (Ex1)
almtry_sum = read.csv("../Tree_allometry/allometry_summary.csv")
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
ggsave(file = "Ex2_TotalDW.pdf", width = 10)
ggsave(file = "Ex2_TotalDW.jpg", width = 10)
show(a1)

# total Height
a2 = ggplot(almtry_sum_cut,aes(x = Tree.., y = H..m.)) +
  geom_bar(stat = "identity") +
  ylab("Tree Height (m)") +
  xlab("") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1),
        panel.background = element_rect(fill = "transparent", colour = "black"),
        text = element_text(size = 32))
ggsave(file = "Ex2_TH.pdf", width = 10)
ggsave(file = "Ex2_TH.jpg", width = 10)
show(a2)

# DBH
a3 = ggplot(almtry_sum_cut,aes(x = Tree.., y = DBH..cm.)) +
  geom_bar(stat = "identity") +
  ylab("DBH (cm)") +
  xlab("") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1),
        panel.background = element_rect(fill = "transparent", colour = "black"),
        text = element_text(size = 32))
ggsave(file = "Ex2_DBH.pdf", width = 10)
ggsave(file = "Ex2_DBH.jpg", width = 10)
show(a3)

# Total leaf area
a4 = ggplot(almtry_sum_cut,aes(x = Tree.., y = Total.leaf.area..m2.)) +
  geom_bar(stat = "identity") +
  ylab("Leaf area (m^2)") +
  xlab("") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1),
        panel.background = element_rect(fill = "transparent", colour = "black"),
        text = element_text(size = 32))
ggsave(file = "Ex2_LA.pdf", width = 10)
ggsave(file = "Ex2_LA.jpg", width = 10)
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
ggsave(file = "Ex2_DW_pct.pdf", width = 10)
ggsave(file = "Ex2_DW_pct.jpg", width = 10)
show(a5)

# branch profile
bra = read.csv("../Tree_allometry/branch_drymass_proflie.csv")
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
pdf("EX2_BranchProfile.pdf",width = 12)
multiplot(plotlist = b, cols = 8)
dev.off()
pdf("EX2_BranchProfile.jpg",width = 1000,height=600)
multiplot(plotlist = b, cols = 8)
dev.off()


# leaf mass profile
lm = read.csv("../Tree_allometry/Leafmass_profile.csv")
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
pdf("EX2_LeafmassProfile.pdf",width = 12)
multiplot(plotlist = b, cols = 8)
dev.off()
jpeg("EX2_LeafmassProfile.jpg",,width = 1000,height=600)
multiplot(plotlist = b, cols = 8)
dev.off()

# leaf area profile
lm = read.csv("../Tree_allometry/Leafarea_profile.csv")
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
pdf("EX2_LeafareaProfile.pdf",width = 12)
multiplot(plotlist = b, cols = 8)
dev.off()
jpeg("EX2_LeafareaProfile.jpg",width = 1000,height=600)
multiplot(plotlist = b, cols = 8)
dev.off()

# stem mass profile
lm = read.csv("../Tree_allometry/stem_drymass_profile.csv")
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
pdf("EX2_StemmassProfile.pdf",width = 12)
multiplot(plotlist = b, cols = 8)
dev.off()
jpeg("EX2_StemmassProfile.jpg",width = 1000,height=600)
multiplot(plotlist = b, cols = 8)
dev.off()

# stem diameter profile
sm = read.csv("../Tree_allometry/stem_diameter_profile.csv")
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
pdf("EX2_StemdiameterProfile.pdf",width = 12)
multiplot(plotlist = b, cols = 8)
dev.off()
jpeg("EX2_StemdiameterProfile.jpg",,width = 1000,height=600)
multiplot(plotlist = b, cols = 8)
dev.off()


############################################################
# Gap fraction measuremesnts
############################################################
gap = read.csv("../OverstoryLAI/GapFraction_20180724_for_PAIcalc.csv")

c1 <- ggplot(gap, aes(x=Path_num*10, y=Ring.1)) + 
  geom_point(size = 1.5, aes(shape=TransectID, color=TransectID)) +
  geom_line(aes(linetype= TransectID, color=TransectID), lwd =0.5) +
  ylim(0,1) +
  ylab("Gap fraction") +
  xlab("Transect line (m)") +
  ggtitle("Ring 1") +
  theme(panel.background = element_rect(fill = "transparent", colour = "black", size = 1.1),
        text = element_text(size = 14,color="black"),  axis.text = element_text(size = 14,color="black"),
        legend.position = "bottom",
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank())

c2 <- ggplot(gap, aes(x=Path_num*10, y=Ring.2)) + 
  geom_point(size = 1.5, aes(shape=TransectID, color=TransectID)) +
  geom_line(aes(linetype= TransectID, color=TransectID), lwd =0.5) +
  ylim(0,1) +
  ylab("Gap fraction") +
  xlab("Transect line (m)") +
  ggtitle("Ring 2") +
  theme(panel.background = element_rect(fill = "transparent", colour = "black", size = 1.1),
        text = element_text(size = 14,color="black"),  axis.text = element_text(size = 14,color="black"),
        legend.position = "bottom",
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank())

c3 <- ggplot(gap, aes(x=Path_num*10, y=Ring.3)) + 
  geom_point(size = 1.5, aes(shape=TransectID, color=TransectID)) +
  geom_line(aes(linetype= TransectID, color=TransectID), lwd =0.5) +
  ylim(0,1) +
  ylab("Gap fraction") +
  xlab("Transect line (m)") +
  ggtitle("Ring 3") +
  theme(panel.background = element_rect(fill = "transparent", colour = "black", size = 1.1),
        text = element_text(size = 14,color="black"),  axis.text = element_text(size = 14,color="black"),
        legend.position = "bottom",
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank())

c4 <- ggplot(gap, aes(x=Path_num*10, y=Ring.4)) + 
  geom_point(size = 1.5, aes(shape=TransectID, color=TransectID)) +
  geom_line(aes(linetype= TransectID, color=TransectID), lwd =0.5) +
  ylim(0,1) +
  ylab("Gap fraction") +
  xlab("Transect line (m)") +
  ggtitle("Ring 4") +
  theme(panel.background = element_rect(fill = "transparent", colour = "black", size = 1.1),
        text = element_text(size = 14,color="black"),  axis.text = element_text(size = 14,color="black"),
        legend.position = "bottom",
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank())

c5 <- ggplot(gap, aes(x=Path_num*10, y=Ring.5)) + 
  geom_point(size = 1.5, aes(shape=TransectID, color=TransectID)) +
  geom_line(aes(linetype= TransectID, color=TransectID), lwd =0.5) +
  ylim(0,1) +
  ylab("Gap fraction") +
  xlab("Transect line (m)") +
  ggtitle("Ring 5") +
  theme(panel.background = element_rect(fill = "transparent", colour = "black", size = 1.1),
        text = element_text(size = 14,color="black"),  axis.text = element_text(size = 14,color="black"),
        legend.position = "bottom",
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank())

pdf("Ex3_GapFraction_Ring.pdf",width = 12)
multiplot(plotlist = list(c1,c2,c3,c4,c5), cols = 3)
dev.off()
jpeg("Ex3_GapFraction_Ring.jpg",width = 1000,height=600)
multiplot(plotlist = list(c1,c2,c3,c4,c5), cols = 3)
dev.off()

############################################################
# Understory LAI
############################################################

ulai = read.csv("../Understory_LAI_biomass/Understory_LAI_biomass.csv")
ulai_sub = ulai[-17,]
ulai_odr =transform(ulai_sub, SiteID = 
                      factor(SiteID, levels = c("U-0", "U-1", "U-2", "U-3", "U-4", 
                                                "U-5", "U-6", "U-7", "U-8", "U-9",
                                                "U-10", "U-11", "U-12", "U-13", 
                                                "U-14", "U-15")))

# LAI bar plot
e1 = ggplot(ulai_odr,aes(x = SiteID, y = LAI..m2.m.2.)) +
  geom_bar(stat = "identity") +
  geom_hline(aes(yintercept=mean(ulai_odr$LAI..m2.m.2,na.rm=TRUE)),
             color="blue", linetype="dashed", size=1) +
  ylab("LAI") +
  xlab("Site ID") +
  ggtitle(paste("LAI mean =",mean(ulai_odr$LAI..m2.m.2,na.rm=TRUE),sep =" ")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1,color="black"),
        panel.background = element_rect(fill = "transparent", colour = "black",size=1),
        text = element_text(size = 24),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank())
ggsave(file = "Ex4_uLAI.pdf", width = 10)
ggsave(file = "Ex4_uLAI.jpg", width = 10)
show(e1)

# above ground leaf and woody biomass

cn = c("SiteID","Total_Leaf_Area","Ground.area..m.2." ,"LAI..m2.m.2.",
       "LeafBiomass","WoodyBiomass","Total_above_ground_biomass",
       "Cloudberry_leaf_biomass","Cotton_Grass_leaf_biomass",
       "Bilberry_leaf_biomass","Cranberry_leaf_biomass",
       "BogBirch_leaf_biomass", "BlackSpruce_leaf_biomass",
       "LabradorTea_leaf_biomass","Others_leaf_biomass",
       "Cloudberry_woody_biomass","Cotton_Grass_woody_biomass",
       "Bilberry_woody_biomass","Cranberry_woody_biomass","BogBirch_woody_biomass",     
       "BlackSpruce_woody_biomass","LabradorTea_woody_biomass", 
       "Others_woody_biomass","Dry_Material_biomass")
colnames(ulai_odr) = cn

# mass partitions
temp1 = melt(ulai_odr, id.vars ="SiteID") 
temp2 = subset(temp1,temp1$variable =="LeafBiomass" 
               | temp1$variable =="WoodyBiomass")
temp2$value = as.numeric(temp2$value)

# average fraction of foliage biomass fraction to woody
ff= 100*ulai$Leaf_Biomass..g...m.2.[17]/(ulai$Leaf_Biomass..g...m.2.[17]+ulai$WoodyBiomass..g.m.2.[17])

e2 = ggplot(temp2,aes(x = SiteID, y = value, fill=variable)) +
  geom_bar(stat = "identity") +
  scale_fill_brewer(palette='Set1') +
  ylab("Above ground dry mass (g)") +
  xlab("Tree ID") +
  ggtitle(paste("Mean foliage fraction =",floor(100*ff)/100,"%",sep =" ")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        panel.background = element_rect(fill = "transparent", colour = "black",size=1),
        text = element_text(size = 24),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        legend.position = c(0.8,0.9))
ggsave(file = "Ex4_AboveBiomass.pdf", width = 10)
ggsave(file = "Ex4_AboveBiomass.jpg", width = 10)
show(e2)

# mass partitions
# total 
Total = ulai[17,]
colnames(Total) = cn
ptemp1 = melt(Total, id.vars ="SiteID") 

# Foliage/woody portion　biomass
v= grep("leaf", ptemp1$variable ) 
ptemp2= ptemp1[v,]
ptemp2$SiteID = "Foliage"
v= grep("woody", ptemp1$variable ) 
ptemp3= ptemp1[v,]
ptemp3$SiteID = "Woody"
ptemp5 = rbind(ptemp2,ptemp3)

cn = c("Cloudberry","Cotton","Bilberry","Cranberry","BogBirch","BlackSpruce","LabradorTea","Others",
       "Cloudberry","Cotton","Bilberry","Cranberry","BogBirch","BlackSpruce","LabradorTea","Others")

ptemp5[,2] = cn

e3 = ggplot(ptemp5,aes(x = SiteID, y = value, fill=variable)) +
  geom_bar(stat = "identity", position = "fill") +
  scale_fill_brewer(palette='Set1') +
  ylab("Dry mass fraction") +
  xlab("") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        panel.background = element_rect(fill = "transparent", colour = "black",size=1),
        text = element_text(size = 24),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank())
ggsave(file = "Ex4_Summary_AvobeBiomassPortion.pdf", width = 10)
ggsave(file = "Ex4_Summary_AvobeBiomassPortion.jpg", width = 10)
show(e3)

# Foliage portion biomass for each Site ID
qtemp1 = melt(ulai_odr, id.vars ="SiteID") 

# Foliage/woody portion　biomass
v= grep("leaf", qtemp1$variable ) 
qtemp2= qtemp1[v,]
#qtemp2 = cbind(qtemp2,"Foliage")

cn=c(1:128)
cn[1:16] = "Cloudberry"
cn[17:32] = "Cotton"
cn[33:48] = "Bilberry"
cn[49:64] = "Cranberry"
cn[65:80] = "BogBirch"
cn[81:96] = "BlackSpruce"
cn[97:112] = "LabradorTea"
cn[113:128] = "Others"

qtemp2[,2] = cn
e4 = ggplot(qtemp2,aes(x = SiteID, y = value, fill=variable)) +
  geom_bar(stat = "identity", position = "fill") +
  scale_fill_brewer(palette='Set1') +
  ylab("Foliage dry mass") +
  xlab("") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        panel.background = element_rect(fill = "transparent", colour = "black",size=1),
        text = element_text(size = 24),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank())
ggsave(file = "Ex4_FoliageBiomass.pdf", width = 10)
ggsave(file = "Ex4_FoliageBiomass.jpg", width = 10)
show(e4)

# woody portion　biomass
v= grep("woody", qtemp1$variable) 
qtemp2= qtemp1[v,]

cn=c(1:128)
cn[1:16] = "Cloudberry"
cn[17:32] = "Cotton"
cn[33:48] = "Bilberry"
cn[49:64] = "Cranberry"
cn[65:80] = "BogBirch"
cn[81:96] = "BlackSpruce"
cn[97:112] = "LabradorTea"
cn[113:128] = "Others"

qtemp2[,2] = cn
e5 = ggplot(qtemp2,aes(x = SiteID, y = value, fill=variable)) +
  geom_bar(stat = "identity", position = "fill") +
  scale_fill_brewer(palette='Set1') +
  ylab("Woody dry mass") +
  xlab("") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        panel.background = element_rect(fill = "transparent", colour = "black",size=1),
        text = element_text(size = 24),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank())
ggsave(file = "Ex4_WoodyBiomass.pdf", width = 10)
ggsave(file = "Ex4_WoodyBiomass.jpg", width = 10)
show(e5)

## LMA summary from July 2018 
lma = read.csv("../Understory_LAI_biomass/LMA.csv")

e6 = ggplot(lma,aes(x = Species, y = LMA.g.m.2.)) +
  geom_bar(stat = "identity") +
  ylab("LMA (g / m^2)") +
  xlab(" ") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1,colour = "black"),
        panel.background = element_rect(fill = "transparent", colour = "black"),
        text = element_text(size = 24))
ggsave(file = "Ex4_LMA.pdf", width = 8)
ggsave(file = "Ex4_LMA.jpg", width = 8)
show(e6)


