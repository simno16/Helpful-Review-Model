# Helpful Review Prediction With XGBoost and SVM

# 1. Load Libraries
packages<-c("jsonlite", "dplyr", "text2vec", "xgboost", "pROC", "Matrix", "LiblineaR", "ggplot2")
for(p in packages){
  if(!require(p, character.only=TRUE)) install.packages(p)
  library(p, character.only=TRUE)
}

# 2. Load JSONL file 
temporary<-file("All_Beauty.jsonl", open="r")
reviews<-jsonlite::stream_in(temporary, verbose=FALSE)
close(temporary)

# 3. Prepare data
df<-reviews%>%
  select(text, helpful_vote)%>%
  filter(!is.na(text))%>%
  mutate(helpful_flag=helpful_vote>0,
         review_length=nchar(text))  

set.seed(123)
  df_sample<-df%>%sample_n(80000)
  
# Train/test split
set.seed(123)
train_idx<-sample(seq_len(nrow(df_sample)), size=0.8*nrow(df_sample))
df_train<-df_sample[train_idx, ]
df_test<-df_sample[-train_idx, ]

# 4. Text pre-processing
library(text2vec)
prep_fun<-tolower
tok_fun<-word_tokenizer

it_train<-itoken(df_train$text, preprocessor=prep_fun, tokenizer=tok_fun, progressbar=TRUE)
it_test<-itoken(df_test$text, preprocessor=prep_fun, tokenizer=tok_fun, progressbar=TRUE)

#5. Graphs
library(dplyr)
library(ggplot2)

# Classify "helpful"
df_sample<-df_sample%>%
  mutate(helpful_flag=ifelse(helpful_vote>0, 1, 0))

# Class distribution
table(df_sample$helpful_flag)
prop.table(table(df_sample$helpful_flag))

# Distribution plot
df_sample%>%
  mutate(helpful_flag=factor(helpful_flag, labels=c("Not Helpful", "Helpful")))%>%
  ggplot(aes(x=helpful_flag, fill=helpful_flag))+
  geom_bar()+
  theme_minimal()+
  labs(title="Distribution of Helpful vs Not Helpful Reviews",
       x="Review Type", y="Count")

# Review length by helpfulness
df_sample<-df_sample%>%
  mutate(review_length=nchar(text))

ggplot(df_sample, aes(x=review_length, fill=factor(helpful_flag)))+
  geom_density(alpha=0.5)+
  theme_minimal()+
  labs(title="Review Length by Helpfulness",
       x="Review Length (characters)", y="Density", fill="Helpful Flag")

# 6. Create TF–IDF matrix
vocab<-create_vocabulary(it_train)
vocab<-prune_vocabulary(vocab, term_count_min=5)
vectorizer<-vocab_vectorizer(vocab)
dtm_train<-create_dtm(it_train, vectorizer)
dtm_test<-create_dtm(it_test, vectorizer)

tfidf<-TfIdf$new()
dtm_train_tfidf<-cbind(dtm_train, df_train$review_length)
dtm_test_tfidf<-cbind(dtm_test, df_test$review_length)


# Add review length as feature
dtm_train_tfidf<-cbind(dtm_train_tfidf, df_train$review_length)
dtm_test_tfidf<-cbind(dtm_test_tfidf, df_test$review_length)

# Labels
y_train<-as.numeric(df_train$helpful_flag)
y_test<-as.numeric(df_test$helpful_flag)

# 7. Train SVM
X_svm_train<-as.matrix(dtm_train_tfidf)
X_svm_test<-as.matrix(dtm_test_tfidf)

svm_model<-LiblineaR(data=X_svm_train, target=y_train, type=1, cost=1, bias=TRUE)

# 8. Predict SVM
pred_svm<-predict(svm_model, X_svm_test)$predictions
pred_svm_class<-ifelse(pred_svm>0, 1, 0)

# Evaluate SVM
conf_matrix_svm<-table(Predicted=pred_svm_class, Actual=y_test)
print(conf_matrix_svm)

precision_svm<-sum(pred_svm_class&y_test)/sum(pred_svm_class)
recall_svm<-sum(pred_svm_class&y_test)/sum(y_test)
f1_svm<-2*precision_svm*recall_svm/(precision_svm+recall_svm)
auc_svm<-pROC::roc(y_test, as.vector(pred_svm))$auc

message("SVM Precision: ", round(precision_svm, 3))
message("SVM Recall: ", round(recall_svm, 3))
message("SVM F1: ", round(f1_svm, 3))
message("SVM AUC: ", round(auc_svm, 3))

# 9. Train XGBoost
total_positive<-sum(y_train==1)
total_negative<-sum(y_train==0)
scale_pos_weight<-total_negative/total_positive

dtrain<-xgb.DMatrix(data=dtm_train_tfidf, label=y_train)
dtest<-xgb.DMatrix(data=dtm_test_tfidf, label=y_test)

params<-list(
  objective="binary:logistic",
  eval_metric="auc",
  max_depth=6,
  eta=0.1,
  scale_pos_weight=scale_pos_weight
)

xgb_model<-xgb.train(
  params=params,
  data=dtrain,
  nrounds=150,
  verbose=1
)

# 10. F1-optimal threshold (using train set only)
pred_train_prob<-predict(xgb_model, dtrain)
thresholds<-seq(0, 1, 0.01)

f1_scores<-sapply(thresholds, function(t){
  p<-ifelse(pred_train_prob>t, 1, 0)
  precision<-sum(p & y_train)/sum(p)
  recall<-sum(p & y_train)/sum(y_train)
  if(is.na(precision)|is.na(recall)) return(0)
  2*precision*recall/(precision+recall)
})

best_threshold<-thresholds[which.max(f1_scores)]
message("F1-optimal threshold: ", best_threshold)

# 11. Predict on test set using XGBoost
pred_test_prob<-predict(xgb_model, dtest)
pred_test_class<-ifelse(pred_test_prob>best_threshold, 1, 0)

# Evaluate XGBoost
conf_matrix_xgb<-table(Predicted=pred_test_class, Actual=y_test)
print(conf_matrix_xgb)

precision_xgb<-sum(pred_test_class&y_test)/sum(pred_test_class)
recall_xgb<-sum(pred_test_class&y_test)/sum(y_test)
f1_xgb<-2*precision_xgb*recall_xgb/(precision_xgb+recall_xgb)
auc_xgb<-pROC::roc(y_test, pred_test_prob)$auc

message("XGBoost Precision: ", round(precision_xgb,3))
message("XGBoost Recall: ", round(recall_xgb,3))
message("XGBoost F1: ", round(f1_xgb,3))
message("XGBoost AUC: ", round(auc_xgb,3))

# 12. Save models
saveRDS(svm_model, "helpful_svm_model.rds")
saveRDS(xgb_model, "helpful_xgb_model.rds")
saveRDS(tfidf, "tfidf_transformer.rds")
saveRDS(vocab, "vocab_info.rds")

message("All models and transformers saved.")
