from transformers import GPTNeoForCausalLM, GPTNeoModel, GPTNeoPreTrainedModel
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from typing import Optional, Tuple, Union
from model.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

import torch
import torch.nn as nn

from transformers.modeling_outputs import CausalLMOutputWithPast

# logger = logging.get_logger(__name__)

class GPTNeoForMTG(GPTNeoPreTrainedModel):
    def __init__(self, config, 
                 tokenizer_length, 
                 card_vocab,
                 ):
        super().__init__(config)
        
        self.create_card_T(config)
        self.transformer = GPTNeoModel.from_pretrained(config._name_or_path)
        for param in self.transformer.parameters():
            param.requires_grad = False
        # Freeze all the parameters in the model

        self.lm_head = nn.Linear(config.hidden_size, card_vocab, bias=False)

        self.tokenizer_length = tokenizer_length
        self.hidden_size = config.hidden_size
        self.card_vocab = card_vocab

        self.resize_token_embeddings(self.tokenizer_length)
        self.post_init()

    def create_card_T(self, config):
        self.card_transformer = GPTNeoModelMTG(config)
        self.card_transformer.h = torch.nn.ModuleList(self.card_transformer.h[6:])
        del self.card_transformer.wte
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        gt: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
    ):

        batch, num_cards, max_length = input_ids.shape
        input_ids = input_ids.view(batch * num_cards, max_length)
        attention_mask = attention_mask.view(batch * num_cards, max_length)

        with torch.no_grad():
            transformer_outputs = self.transformer(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        card_encodings = transformer_outputs[0][:, -1] # Grabs the last embedding
        card_encodings = card_encodings.view(batch, num_cards, self.hidden_size)
        label_encodings = card_encodings[mask].view(batch, -1, self.hidden_size)
        card_input = card_encodings[~mask].view(batch, -1, self.hidden_size)

        loss = 0.0
        all_logits = []
        all_selections = []
        original_labels = labels
        for _ in range(sum(mask[0])):

            hidden_states = self.card_transformer(
                card_input,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            lm_logits = self.lm_head(hidden_states[:, -1]) # torch.Size([12, 12698])
            # Gather the logits related to the grouth truth for teacher forcing
            
            if labels is not None:
            
                next_cards, label_selections, selected_labels = self.get_next_cards(lm_logits, labels, label_encodings)
                # Append those cards to the input
                card_input = torch.cat((card_input, next_cards), dim=1)

                # move labels to correct device to enable model parallelism
                labels = labels.to(lm_logits.device)
                lm_logits = lm_logits.to(torch.float32)

                # All possible choices for model to make
                target = torch.zeros_like(lm_logits).scatter_(1, labels, 1)
                
                # Determine the positive weight value
                num_positives = target[0].sum()
                num_negatives = target[0].numel() - num_positives
                pos_weight_value = num_negatives / num_positives / 10

                loss_fct = BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=lm_logits.device))
                loss += loss_fct(lm_logits, target)

                labels = self.remove_label(labels, label_selections)

                all_selections.append(selected_labels)

            all_logits.append(lm_logits)
                
        if labels is not None:
            all_selections = torch.concatenate(all_selections, dim=1)

        all_logits = torch.stack(all_logits)
        
        if not return_dict:
            output = (all_logits,) + (all_selections,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
    
    def get_next_cards(self, lm_logits, labels, label_encodings):
        label_logits = torch.gather(lm_logits, 1, labels) # torch.Size([12, 10])
        # Select the highest likelihood grouth truth to insert to input
        label_selections = torch.argmax(label_logits, 1).unsqueeze(-1) # torch.Size([12,])
        # Select those labels from labels
        selected_labels = torch.gather(labels, 1, label_selections)
        # Get shape correct for selection of embedded cards
        label_selections_expanded = label_selections.unsqueeze(-1).expand(-1, -1, label_encodings.size(2))
        # Gather those cards from each batch in the gt_encodings
        next_cards = torch.gather(label_encodings, 1, label_selections_expanded)

        return next_cards, label_selections, selected_labels

    def remove_label(self, labels, label_selections):
        # Create a range tensor that matches the second dimension of gt
        range_tensor = torch.arange(labels.size(1), device=labels.device).unsqueeze(0)
        # Broadcast gt_selections to match the shape of gt
        broadcasted_gt_selections = label_selections.expand(-1, labels.size(1))
        # Create a mask where the index matches the gt_selections index
        mask = range_tensor != broadcasted_gt_selections
        # Apply the mask to gt to keep only the values that do not match gt_selections
        labels = labels[mask].view(labels.size(0), -1)

        return labels

class DummyWTE(nn.Module):
    def __init__(self):
        super(DummyWTE, self).__init__()

    def forward(self, input):
        return input
    
class GPTNeoModelMTG(GPTNeoModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()[:-1]
            # input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)

        inputs_embeds = input_ids
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        # Attention mask.
        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(attention_mask, input_shape, inputs_embeds, past_length)

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                # logger.warning_once(
                #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                # )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return hidden_states