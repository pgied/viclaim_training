
import math


def load_and_filter_data_for_sentence_only(data, annotator_map, weight_by_topic=False, do_soft=False):
    dataset = []
    multi_label_count = 3
    single_label_count = math.pow(2, multi_label_count)

    for entry in data:
        language = entry['language']
        topic = entry['topic']
        clip_id = entry['clip_id']
                
        # Concatenate all sentences into one text
        full_text = " ".join(entry['sentences'].values())    
        sentence_entity_list = entry['sentence_entity_list']
        # for se_key, se in sentence_entity_list.items(): 
        #     se_id = se['se_id']

        #     if not do_soft:
        #         accumulated_single_label = [0] * single_label_count
        #         calculated_single_label = [0] * single_label_count

        #         for annotation in se['annotation_list']:
        #             if not annotation['is_gold_annotator']:
        #                 single_label_index = convert_to_single_label_format(annotation['labels'])
        #                 if weight_by_topic:
        #                     accumulated_single_label[single_label_index] += annotator_map[annotation['annotator_id']]['weights'][topic]
        #                 else:
        #                     accumulated_single_label[single_label_index] += annotator_map[annotation['annotator_id']]['weights']['general']

        #         calculated_single_label[max(enumerate(accumulated_single_label), key=lambda x: x[1])[0]] = 1
        #         multi_label = convert_single_to_multi_label(calculated_single_label)
        #     else:
        #         annotations = se['annotation_list']
        #         labels = np.array([annotation['labels'] for annotation in annotations if not annotation['is_gold_annotator']])
        #         if weight_by_topic:
        #             ann_weights = np.array( [annotator_map[annotation['annotator_id']]['weights'][topic] for annotation in annotations if not annotation['is_gold_annotator']])
        #         else:
        #             ann_weights = np.array( [annotator_map[annotation['annotator_id']]['weights']['general'] for annotation in annotations if not annotation['is_gold_annotator']])
        #         multi_label = (labels.T @ softmax(ann_weights)).tolist()
        #         single_label_labels_ind = np.array([convert_to_single_label_format(annotation['labels']) for annotation in annotations if not annotation['is_gold_annotator']])
        #         single_label_labels = np.eye(8)[single_label_labels_ind]
        #         calculated_single_label = (single_label_labels.T @ softmax(ann_weights)).tolist()


        #     item = {
        #         'clip_id': clip_id,
        #         'se_id': se_id,
        #         'language': language,
        #         'topic': topic,
        #         'full_text': full_text,
        #         'sentence': se['sentence'],
        #         'single_label': calculated_single_label,
        #         'multi_label': multi_label
        #     }

        #     dataset.append(item)
            
    return dataset