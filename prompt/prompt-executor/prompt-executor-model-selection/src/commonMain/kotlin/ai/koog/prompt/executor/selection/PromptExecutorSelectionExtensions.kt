package ai.koog.prompt.executor.selection

import ai.koog.agents.core.tools.ToolDescriptor
import ai.koog.prompt.Prompt
import ai.koog.prompt.dsl.ModerationResult
import ai.koog.prompt.executor.model.PromptExecutor
import ai.koog.prompt.llm.LLModel
import ai.koog.prompt.message.LLMChoice
import ai.koog.prompt.message.Message
import ai.koog.prompt.streaming.StreamFrame
import io.github.oshai.kotlinlogging.KLogger
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.emitAll
import kotlinx.coroutines.flow.flow

public sealed interface SelectionExecutionPolicy

public object TryBest : SelectionExecutionPolicy

public open class TryUpTo(public val maxModelsToTry: Int) : SelectionExecutionPolicy

public object TryAll : SelectionExecutionPolicy

public class SelectionExecutionException : Exception()

public object PromptExecutorExtensions {

    private val logger = KotlinLogging.logger { }

    /**
     * Executes [prompt] using the highest-ranked model from [selection].
     */
    public suspend fun PromptExecutor.execute(
        prompt: Prompt,
        tools: List<ToolDescriptor> = emptyList(),
        selectionExecutionPolicy: SelectionExecutionPolicy = TryBest,
        selection: DefaultModelSelectorBuilder.() -> Unit,
    ): Message.Assistant {
        val selector = DefaultModelSelectorBuilder().apply(selection).build()
        val modelsToAttempt = selector.select(models()).modelsToAttempt(selectionExecutionPolicy)
        modelsToAttempt.forEachIndexed { attempt, model ->
            try {
                return execute(prompt, model, tools)
            } catch (e: Exception) {
                logger.info(e) { "Selection based execution failed (attempt: ${attempt + 1}, model: $model)" }
            }
        }
        throw SelectionExecutionException("All models failed to execute prompt", failures)
    }

    private fun ModelSelection.modelsToAttempt(selectionExecutionPolicy: SelectionExecutionPolicy): List<LLModel> {
        if (ranked.isEmpty()) throw IllegalStateException("No models available for selection")
        return when (selectionExecutionPolicy) {
            is TryBest -> listOf(ranked.first())
            is TryUpTo -> ranked.take(selectionExecutionPolicy.maxModelsToTry)
            is TryAll -> ranked
        }
    }

    /**
     * Streams output frames for [prompt] using the highest-ranked model from [selection].
     */
    public fun PromptExecutor.executeStreaming(
        prompt: Prompt,
        tools: List<ToolDescriptor> = emptyList(),
        selection: DefaultModelSelectorBuilder.() -> Unit,
    ): Flow<StreamFrame> = flow {
        val model = selectSingleModel(DefaultModelSelectorBuilder().apply(selection).build())
        emitAll(executeStreaming(prompt, model, tools))
    }

    /**
     * Returns multiple independent choices for [prompt] using the highest-ranked model from [selection].
     */
    public suspend fun PromptExecutor.executeMultipleChoices(
        prompt: Prompt,
        tools: List<ToolDescriptor> = emptyList(),
        selection: DefaultModelSelectorBuilder.() -> Unit,
    ): LLMChoice {
        val model = selectSingleModel(DefaultModelSelectorBuilder().apply(selection).build())
        return executeMultipleChoices(prompt, model, tools)
    }

    public suspend fun PromptExecutor.moderate(
        prompt: Prompt,
        selection: DefaultModelSelectorBuilder.() -> Unit,
    ): ModerationResult {
        val model = selectSingleModel(DefaultModelSelectorBuilder().apply(selection).build())
        return moderate(prompt, model)
    }
}

private suspend fun PromptExecutor.selectSingleModel(modelSelector: ModelSelector): LLModel {
    val selection = modelSelector.select(models())
    return selection.ranked.firstOrNull()
        ?: throw IllegalArgumentException("No model matched selection criteria")
}
