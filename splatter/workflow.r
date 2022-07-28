library("splatter")
library("scater")

de_prob = 0.3
sim <- splatSimulateGroups(nGenes=1000, batchCells = 6000, de.prob = de_prob, group.prob = c(0.5, 0.5), de.facLoc = 0.01, verbose = FALSE)
counts <- counts(sim)
types = colData(sim)

write.csv(counts, cat(cat("counts_splatter_de.prob=", de_prob, sep = ""), ".csv", sep=""))
write.csv(types,cat(cat("types_splatter_de.prob=", de_prob, sep = ""), ".csv", sep=""))



sim <- logNormCounts(sim)
sim <- runPCA(sim, ncomponents = 2)

jpeg(cat(cat("PCAplot_de.prob=", de_prob, sep = ""), ".jpg", sep=""))
plotPCA(sim, colour_by = "Group") 
dev.off()