library(ScottKnottESD)
library(readr)
library(ggplot2)
library(RColorBrewer)
library(dplyr)

#sk-ESD for feature importance for merged/abandoned effort aware prediction
ORGS = c('Libreoffice','Gerrithub','Eclipse')
DATA_PATH = 'C:/Users/Motaz/Desktop/work/prepare_tex_tables/CEC2023'
ALGO_LIST = c('NSGA2_LR_AUC_average_churn_recall')
METRICS = c('popt','recall_at_0.2','auc')
FEATURES = c('author_experience','author_merge_ratio', 'author_changes_per_week',
             'author_merge_ratio_in_project', 'total_change_num',
             'author_review_num', 'description_length', 'is_documentation',
             'is_bug_fixing', 'is_feature', 'project_changes_per_week',
             'project_merge_ratio', 'changes_per_author', 'num_of_reviewers',
             'num_of_bot_reviewers', 'avg_reviewer_experience',
             'avg_reviewer_review_count', 'lines_added', 'lines_deleted',
             'files_added', 'files_deleted', 'files_modified', 'num_of_directory',
             'modify_entropy', 'subsystem_num')

compare_features <- function(orgs,data,features,algos,greater_is_better = TRUE){
  results <- data.frame(Organization = character(),model = character(),group = double(),mean = double(),median = double())
  for (org_name in orgs) {
    cat('working on org',org_name,'\n')
    org_data = subset(data,Project == org_name)
    cat(features)
    algos_list = NULL
    for (algo in algos) {
      cat('Working on model',algo,'\n')
      algo_data <- subset(org_data,Approach==algo)
      if (greater_is_better == FALSE){
        algo_data <- algo_data*-1
      }
      metric_df =  algo_data[features]
      metric_df %>% filter_all(any_vars(is.numeric(.) & . < 2000))
      sk <- sk_esd(metric_df)
      sk_ranks <- data.frame(Feature = names(sk$groups),
                             rank = paste0('Rank-', sk$groups))
      plot(sk)
      print(names(sk$groups))
      #print(sk_ranks$rank)
      plot_data <- melt(metric_df)
      #print(plot_data)
      plot_data <- merge(plot_data, sk_ranks, by.x = 'variable', by.y = 'Feature')
      nb.cols <- 18
      mycolors <- colorRampPalette(brewer.pal(8, "Set2"))(nb.cols)
      #print(plot_data)
      #vals <- as.numeric(gsub("Rank-","", plot_data$rank))
      #print(vals)
      #print('===============')
      #print(order(vals))
      #plot_data <- plot_data[order(vals), ]
      #print(plot_data)
      unique_ranks <- unique(plot_data$rank)
      unique_ranks<- unique_ranks[order(nchar(unique_ranks), unique_ranks)]
      print(order(unique(plot_data$rank)))
      plot_data$rank <- factor(plot_data$rank, levels = unique_ranks)
      g <- ggplot(data = plot_data, aes(x = reorder(variable,rank), y = value, fill = rank,middle=mean(variable))) +
        geom_boxplot()  + 
        facet_grid(~rank, scales = 'free_x') +
        scale_fill_manual(values = mycolors) + 
        ylab('Importance') + xlab('Feature') + ggtitle('')  +theme_bw() +
        theme(text = element_text(size = 10),
              legend.position = 'none', axis.text.x = element_text(angle = 90, vjust = 0.5,hjust = 1))
      return(g)      
      
    
}
  }
}
ALL_ORGS_DATA <- read.csv(paste(DATA_PATH,'/','all_features_importance.csv',sep=''))
g = compare_features(orgs=ORGS,data=ALL_ORGS_DATA,features=FEATURES ,algos=ALGO_LIST,greater_is_better = TRUE)