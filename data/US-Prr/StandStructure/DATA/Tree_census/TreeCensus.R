# This script creates various plots/histograms from the original data sets
# Hideki Kobayashi, Oct 6, 2022


library(ggplot2)
library(reshape2)
library(gridExtra)
library(Rmisc)

# Create figures

# Direct sampling of black spruce 

# Tree census measurements (Ex 1)
census = read.csv("TreeCensus.csv")

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
ggsave(file = "Figures/Tree_census_position.pdf", width = 13)
ggsave(file = "Figures/Tree_census_position.jpg", width = 13)

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
ggsave(file = "Figures/DBH_2010.pdf", width = 10)
ggsave(file = "Figures/DBH_2010.jpg", width = 10)

p2=ggplot(census,aes(x = TreeHeight_2010..m.)) +
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
ggsave(file = "Figures/TH_2010.pdf", width = 10)
ggsave(file = "Figures/TH_2010.jpg", width = 10)

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
ggsave(file = "Figures/DBH_2014.pdf", width = 10)
ggsave(file = "Figures/DBH_2014.jpg", width = 10)

p4=ggplot(census,aes(x = TreeHeight_2014..m.)) +
  geom_histogram(aes(y=..density..),color="black", fill="blue") +
  geom_density(alpha=.2, fill="#FF6666") +
  xlim(0,8) +
  ylim(0,0.8) +
  ylab("Density") +
  xlab("Height (m)") +
  geom_vline(aes(xintercept=mean(TreeHeight_2014..m.,na.rm=TRUE)),
             color="blue", linetype="dashed", size=1) +
  theme(
    panel.background = element_rect(fill = "transparent", colour = "black"),
    text = element_text(size = 36))
ggsave(file = "Figures/TH_2014.pdf", width = 10)
ggsave(file = "Figures/TH_2014.jpg", width = 10)
