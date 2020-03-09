library(ggplot2)
library(gridExtra)

args = commandArgs(trailingOnly=TRUE)

if(length(args) == 0) {
  args[1] = "results.csv"
}

print(paste("Reading file", args[1]))

accuracies = read.csv(args[1])

three = accuracies[ accuracies$nr_classes == 3, ]
four = accuracies[ accuracies$nr_classes == 4, ]

# Majority data hardcoded to plot little stars for comparison
majority_three = data.frame("author" = c("a","b","c","d"),
                            "nr_classes" = c(3,3,3,3),
                            "method" = c("majority","majority","majority","majority"),
                            "accuracy" = c(0.40, 0.39, 0.48, 0.42))
majority_four = data.frame("author" = c("a","b","c","d"),
                           "nr_classes" = c(4,4,4,4),
                           "method" = c("majority","majority","majority","majority"),
                           "accuracy" = c(0.43, 0.37, 0.46, 0.44))

three = rbind(three, majority_three)
four = rbind(four, majority_four)

theme = theme(
  panel.grid.major = element_blank(),
  panel.grid.minor = element_blank(),
  panel.background = element_blank(),
  panel.border = element_rect(color = "black", fill = NA, size = 1),
  
  axis.title.x = element_blank(),
  axis.title.y = element_blank(),
  plot.title = element_text(hjust = 0.5),
  
  legend.position = c(.98, .98),
  legend.justification = c("right", "top"),
  legend.title = element_blank(),
  legend.key = element_rect(fill = "white", colour = NA),
  legend.box.background = element_rect(color = "black", size = 0.5)
)

scale_y = scale_y_continuous(limits = c(0.35, 0.8), breaks = seq(0.3, 0.85, 0.05))


three_plot = ggplot() +
  ggtitle("Average accuracies, three-class data") +
  geom_point(data = three, aes(author, accuracy, shape = method, size = method)) +
  geom_rug(data = three, aes(y = accuracy), size = 0.3) +
  scale_shape_manual(values = c(0, 1, 8)) +
  scale_size_manual(values = c(4, 4, 1.5)) +
  scale_y +
  theme

four_plot = ggplot() +
  ggtitle("Average accuracies, four-class data") +
  geom_point(data = four, aes(author, accuracy, shape = method, size = method)) +
  geom_rug(data = four, aes(y = accuracy), size = 0.3) +
  scale_shape_manual(values = c(0, 1, 8)) +
  scale_size_manual(values = c(4, 4, 1.5)) +
  scale_y +
  theme

grid.arrange(three_plot, four_plot, ncol = 2)

g = arrangeGrob(three_plot, four_plot, ncol = 2)
ggsave(file="avg_accuracies.png", g, width = 15, height = 7, scale = 2, units = "cm")

