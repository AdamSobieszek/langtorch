# Playing Games with Ais: The Limits of GPT-3 and Similar Large Language Models

> Adam Sobieszek & Tadeusz Price, 2022

## Abstract

This article contributes to the debate around the abilities of large language models such as GPT-3, dealing with:
firstly, evaluating how well GPT does in the Turing Test, secondly the limits of such models, especially their tendency
to generate falsehoods, and thirdly the social consequences of the problems these models have with truth-telling. We
start by formalising the recently proposed notion of reversible questions, which Floridi & Chiriatti (2020) propose
allow one to ‘identify the nature of the source of their answers’, as a probabilistic measure based on Item Response
Theory from psychometrics. Following a critical assessment of the methodology which led previous scholars to dismiss
GPT’s abilities, we argue against claims that GPT-3 completely lacks semantic ability. Using ideas of compression,
priming, distributional semantics and semantic webs we offer our own theory of the limits of large language models like
GPT-3, and argue that GPT can competently engage in various semantic tasks. The real reason GPT’s answers seem senseless
being that truth-telling is not amongst them. We claim that these kinds of models cannot be forced into producing only
true continuation, but rather to maximise their objective function they strategize to be plausible instead of truthful.
This, we moreover claim, can hijack our intuitive capacity to evaluate the accuracy of its outputs. Finally, we show how
this analysis predicts that a widespread adoption of language generators as tools for writing could result in permanent
pollution of our informational ecosystem with massive amounts of very plausible but often untrue texts.The problem of
learning about identity through questions has recently been taken up by Floridi & Chiriatti (2020) in the context of the
language generator model GPT-3 (Brown et al.,2020). The model can be used to continue any textual prompt given to it
while maintaining a consistent style, correct grammar and will usually make sense. The discussion of answering questions
in the context of computer intelligence dates to the Turing Test and is still widely debated today (Damassino & Novelli,
2020; Montemayor, 2021).

## Introduction

> Are you a performer?
> Yes.
> Could you be considered a leading man?
> Yes.
> Do you have anything to do with sports?
> Yes.
> Would you be considered a writer?
> Yes.
> Do you do any drawings, like comic strips?
> Yes.
> Erm… You are a human being?

These are questions put to Salvador Dali when in 1957 he appeared on the televi- sion programme “What’s my line?”. A
group of blindfolded panellists was tasked with discovering the surrealist’s occupation by asking yes/no questions. In
their article, Floridi and Chiriatti analyse GPT’s ability to respond to questions. They look for the type of questions
which ‘may enable one to identify the nature of the source of their answers’, which they call_reversible_, and at
questions which they deem_irreversible_, meaning we cannot tell from the answers to them whether we are conversing with
a human or a machine. Examples of the latter are mathematical, factual (trivia), and binary (yes/no) questions, while a
potential source of reversible questions are semantic ones such that ‘require understanding […] of both meaning and
context’, especially ones that seem to hinge on real-world experience.

We take issue with some methodological underpinnings of their tests, which led them to dismiss GPT’s abilities. By
amending some of them, we wish to develop further the notion of reversibility, on the grounds that an accurate
highlighting of the obstacles that the current machine learning paradigms face is relevant both to the accurate
assessment of their shortcomings in becoming general AI, as well as their wider social impact.

We’ll show some situations in which reversibility, that is identification of AI-written texts, can come from
counterintuitive sources (that could serve as better guides for identifying AI texts in the future), which suggest there
is not a clear-cut group of semantic questions which will trip up GPT-3. To guide our discussion, we introduce (1.1)
that a response’s information value about the identity of the respondent can be formalized owing to developments in the
field of psychometrics, and that (1.2) GPT’s abilities should be understood as stemming from its learning conditional
probabilities in language (its so-called “statistical capabilities”). It is specifically how far this simple ability can
get GPT that is the crux of the matter, and what shall inform our discussion of its limits, as we claim it is not enough
to equate this ability with the ability to learn syntax.

In the second part of this paper, we propose a theory of the limits and abilities of statistical language models like
GPT-3. A young Leibniz envisioned a machine, that owing to its understanding of language and impartial calculations,
could eventually derive every true statement (Leibniz, 1666). We argue, that in sharp contrast, the regularities these
models exploit, the compression that happens in their architecture, and the way of interacting with them, all contribute
to the statements they produce being modal in their relationship to truth. This means, that it’s not simply that such
models are unable to learn a representation of the real world, it is that they perform a lossy compression of statements
about the real and possible worlds, which bestows upon them a kind of possible-world semantics. We’ll argue, that in
such models truth is indexical to the text at hand, meaning that as the model encounters the word ‘truth’, it cannot
decode its meaning as ‘true in the actual world’ (cf. Lewis, 1986). In simpler terms, this entails that in order to
increase its objective function the model strategizes to be plausible instead of truthful.

In their paper, Floridi and Chiriatti provide an excellent discussion of the social impact and dangers associated with
the expanding role of such language generating models. This point is crucial, as soon we expect language generators to
be in common use among journalists, translators, civil servants, etc. We thus conclude with a derivation of the possible
dangers of widespread adoption of these generators that stem from the present analysis. Through consideration of
psychological inclinations present during the assessment of statements, based primarily on the work of Mercier &
Sperber (2017; Mercier, 2020; Sperber et al., 2010), we show these possible adverse consequences, including the_modal
drift hypothesis_— that because humans are not psychologically equipped to effectively differentiate truth from
plausible falsehoods among texts generated by language models, a mass production of plausible texts may deteriorate the
quality of our informational ecosystem.

## Learning from questioning GPT-3

The logic of Floridi and Chiriatti’s reversibility is that some questions will be harder than others for a machine to
answer, thus these are the questions that an interviewer should ask in order to learn the interlocutor’s identity in a
Turing Test. A binary approach to reversibility suggests that apart from reversible questions, there are also
questions (for maths or trivia questions), which do not help at all in discovering the identity of the respondent –
_irreversible_. What more formal criteria can we use to identify the source from its answers to questions? We will use a
theory from the field of psychometrics to inform our search for these criteria. Psychometrics is the subfield of
psychology concerned with an objective measurement of hidden (latent) traits of respondents by means of questionnaires,
hence the mathematical theories developed to quantify this process are perfectly suited to handle our task of
reconstructing from answers whether a respondent was human.

### The psychometrics of reversibility

The most mature theory quantifying such matters is the Item Response Theory (Embretson & Reise, 2013). Instead of
focusing on properties of questions, IRT seeks to model the way the agent answers questions as being influenced by the
latent trait measured (e.g. that we should expect different answers, from people of different temperament). Thanks to
our knowledge of how agents with different levels of a trait typically answer questions, we can exploit this regularity
and with every consecutive answer reduce our uncertainty as for the value of that latent trait (in our case – whether
they are human). The amount of information gained by coming to know the answer differs between questions (or_items_) and
it is here that IRT advances the crucial concept of item information, what we may call_informativity_and what we shall
discuss as the construct underlying Floridi and Chiriatti’s reversibility.

How does thinking of answers as being influenced by whether the respondent is or is not human help us determine which
questions are reversible? First, we should specify our problem as the task of discovering a dichotomous latent class H (
Bartolucci, 2007; Brzezińska, 2016), that signifies whether the respondent is or is not a human, from the observable
answers. In such a case such a model lets us assess the amount of information gained from the answer (the item
information) quite easily, as it depends on the probability distribution over possible answers to the question,
conditional on a value of the latent class H. We can even calculate this directly: a simple Bayesian calculation of how
our assumed prior belief that it is equally likely that the respondent is (H) or is not (~H) a human changes upon seeing
an answer A, yields:

The absolute size of the change in probability that the respondent is human — the prior probability P(H)= 0.5 — after
seeing the answer depends on how large the difference between P(A|H) and P(A|~H) is, which means that a question is more
informative, the more different the pattern of answers for humans and non-humans (in this case the probability that they
answer A). To practically obtain estimates of these values psychometricians conduct_standardized_empirical studies. The
standardization is what ensures we can treat the subject’s response as their genuine, unbiased answer – by asking each
respondent the same way. Doing this with GPT-3 could present a challenge, which we will discuss in the next section.
What this principle should remind us of is that there is no one universal task of answering questions, as answers are
always influenced by instructions and the task the respondent ends up believing themselves to be doing. If we expect our
respondent to answer in some way, we have to make sure we were understood.

The mathematically minded reader may equate this expected change in belief with the Kullback–Leibler divergence between
the two conditional distributions over answers, additionally embedded into a sentence space to get rid of their
superficial differences (Mulder & van der Linden, 2009; p. 81–84; Conneau et al., 2018 for a discussion of sentence
embeddings), but in simpler terms what we’ve gained is the insight that the expected amount of information about the
identity of the respondent for a particular question is roughly proportional to the difference between the patterns of
answers for human and non-human agents. This includes both being less or more inclined to give particular answers,
which, as we will see, is a symmetry that can be used to expand the scope of reversible questions.

That’s it for established theory. Before subjecting GPT to this quantification, let us quickly motivate why thinking of
answers in terms of conditional probabilities makes sense, which will become more intuitive when we discuss how GPT-3
works. Floridi (2017) rightly points out, that ‘AI is about decoupling successful problem solving from any need to be
intelligent’. Because of this it is important that a metric for comparing the performance of an AI against that of a
human on a task does not judge the process by which the agent arrives at its solution, but only the solution itself,
which is the probability of an answer given the question. This guarantees such a measure would not be quantifying
intelligence, as is sometimes assumed in the Turing Test (for objection see Russell, 2019), but rather be appropriate
for tests identifying AI as the source of the answers, such as those administered by us or Floridi and Chiriatti. IRT
satisfies this criterion, as it has been developed accounting for the human ability to answer at random, so that it
doesn’t inhibit our ability to perform psychometric tests (going as far as to include a “guessing” parameter). Using
informativity as a measure guards us against discounting an AI’s abilities simply because we consider the way in which
it arrived at an answer to be unintelligent.

When trying to reject the null hypothesis that our respondent is human, we are limited in how much information we can
get from a single answer, because humans also practice some odd ways of answering (think Dalí’s interview). It could be
useful to think, after Montemayor (2021), of Turing Testing as a continuous process, with a changing amount of certainty
during the exchange.

### How to question GPT-3?

The theory of informative questions can guide our discussion of subjecting GPT-3 to a Turing Test. We also need to think
about the proper way of putting a question to the model, in a way that would provide standardization analogous to that
achieved with human subjects by controlling the situation of the test. This is not easy with GPT-3, as it has no
knowledge of the context in which it is being used, and the only thing that we can truly control is the prompt we
provide it with. If GPT cannot get to know our intentions, how is it that we can find different ways of prompting the
model to get the relevant answer to our question (e.g. Zhao et al., 2021; Shin et al., 2020; Reynolds & McDonell, 2021)?
The answer requires a deeper understanding of the workings of GPT-3.

GPT-3 is not a chatbot, but a general-purpose, autoregressive language model trained on a wide set of real texts to
continue any given textual prompt (Radford et al., 2019; Brown et al., 2020). What GPT learned during this training is
to predict the conditional probabilities over possible continuations of the text, given the text that went before it.
These possible continuations are encoded as values called tokens, that represent little bites of language such as “Tu”
or “ring”. When GPT continues a text, it picks one of these probable tokens, appends the text with it, and then
recalculates the probability of continuations for the new prior text elongated by that token. This is the autoregressive
quality of GPT-3 that is rarely discussed in its philosophical treatment (see. Figure 1). Two things should be noted
here: (a) many possible strategies of traversing this tree of possibilities exist (e.g. always picking the most probable
token), we take the default strategy of picking the continuations at random with the same probability as predicted by
the model; (b) The exponential increase in the number of possible sentences that stems from this process is enormous (
given GPT-3’s vocabulary of 50,257, the number of different possible outputs becomes greater than the number of atoms of
the universe at a length of just 18 tokens). Some consequences of (b) include that the probability of each single
continuation generated by GPT-3 (with strategy (a)) is very slim and little can be inferred from such a single
continuation. More importantly, in order to produce coherent continuations, GPT-3 must’ have saved into the weights of
its connections a massive amount of information. However, considering the exponential growth of the number of
conditional probabilities that must be stored, this information must first be somehow compressed, and such compression
usually cannot be lossless (Bernstein & Yue, 2021). What we’ll claim is lost during this compression and the
generalizations made during training are the source of both the limitations and supposed intelligence of such models (a
point we will return to in part 2).

Tree of possible continuations. Top: the prompt (the text provided by us for continuation) is marked in bold. GPT’s
continuation is color-coded, representing the conditional probability of this continuation based on the previous text.
Bottom: The probabilities over continuations, from which GPT picks with a weighted random draw. The probabilities in
each step are determined by the choice in the previous step and the entire text before it. Note, that if at any step the
choice would’ve been different, the probabilities in every subsequent step would also be completely different

Knowing this, we can compare a naïve method of asking GPT questions that involves just noting its continuation, given
the question, to a psychometrician searching the internet for an occurrence of her question and noting the words
following the question as her subject’s response. The problem is the lack of what Pearl (Pearl, 2002; Pearl & Mackenzie,
2019) dubbed the “do” operator — we are conditioning on the appearance of the question, but never actually asking it.
Thus, the possible continuations of a question are limited only by the vast range of possible contexts in which a
question may occur in text. So just as we don’t always follow a question with an immediate answer, we cannot be certain
that GPT will. To be able to treat GPT’s continuation as its answer in earnest, we need to deliberately try and increase
the likelihood that GPT’s continuation of the question will be its best answer and not unrelated gibberish.

Recall that we are restricted in our methods of persuading GPT to answer to changing the prompt we provide. We could
just add a phrase like “Answer the question” before the question, motivated by the belief that the occurrence of such
phrases is more often associated with questions being answered in texts. It turns out variations on this simple
procedure prove extremely effective and have become the primary way of interacting with large language models (
Reynolds & McDonell, 2021). It is a method which the creators of GPT-3 called n-shot learning (Brown et al., 2020).
Specifically, it is an example of zero-shot learning in which only a task description is given, for example ‘Translate
from Italian to English: ridere -’. A one-shot prompt would contain a task description and a demonstration, so
‘Translate from Italian to English: piccolo – small, ridere -’, and a few-shot prompt would contain multiple
demonstrations in addition to the task description. These ways of prompting specify our desired continuation by priming
GPT-3 to interpret the context of the request correctly, which makes the continuation as pertinent as possible (see Fig.
2). This insight, on top of providing us with a proper methodology for asking GPT-3 questions, illustrate two points
which will pop up over and over, that: (a) while such language models were trained only to continue the given text, by
skilfully picking the prompt, we are able to make these models perform tasks other than creating random texts; (b) in
picking these tasks we are limited to the tasks for which we can construct a prompt. In stating (a), which is all too
often omitted from discussions of GPT-3, we agree with Montemayor (2021), that GPT’s ability to perform tasks presents a
surprising semblance of linguistic intelligence, specifically, as we state, while having learned only conditional
probabilities. Lastly, anticipating our search for GPT’s limits, we may foreshadow that we will call such tasks that
satisfy (b) – games, as in “games that GPT can play”, which will further generalize the notion of reversibility.

The effect that specifying the prompt better has on the probability of continuations. The question is the one used by
Floridi and Chiriatti to show GPT’s lack of semantic abilities. Left: probabilities of the continuation without
specifying the task and a sample completion, GPT spirals into an irrelevant text. Right: probabilities when the prompt
is written as to encourage question answering; “one” has become the most probable continuation

### Re-evaluating reversible questions

A person wanting to Turing-test GPT-3 would want to know what kinds of questions are going to help them the most in
figuring out their interlocutor’s identity (that is the reversible questions), and which seem useless for this goal (
irreversible). With the insights given by psychometrics and the methodological considerations, we are ready to revisit
Floridi and Chiriatti’s analysis and see if the questions they deemed irreversible (binary, mathematical and trivia)
also have close to zero item information on whether the respondent is human. To discern between these levels, we’ll call
questions with high item information – informative, and keep the original term reversibility for the questions the
authors deemed reversible. To recap, we approximate a question’s informativity by determining the size of the
discrepancy between response patterns of humans and non-humans like GPT-3. Given what we learned about how GPT-3 works,
we should think of this as the difference between conditional probability distributions over possible answers given the
question, and the questions where GPT-3 has a different answering pattern to humans are more informative.

Let’s start with the simple case of binary questions. One might think that upon seeing a ‘yes’ or ‘no’ answer to a
binary question one has gained no information about the author, because either answer could be given by a machine or
human. We’ll argue this is however not the case, illustrated by the fact that Salvador Dalí’s answers to binary
questions led the panellists on ‘What’s my line?’ to doubt his humanity. When one is dealing either with a human or a
simple machine that answered “yes” to every binary question, like Dalí in the opening paragraph, one should clearly
expect to gain more information about the humanity of the respondent, from questions that more people would be inclined
to answer “no” to. Here, we clearly see why the question would be more informative from the point of view of IRT, as
such questions have a larger impact on our certainty about the value of the latent trait “being human”. There is however
nothing stopping us from extending this notion to sophisticated models like GPT-3, as the only difference there is a
more sophisticated probability distribution over the “yes” and “no” answers. Regardless, the principle still holds that
it is not merely that all binary questions are uninformative, but that the more deviant the distributions over answers
for humans and GPT-3, the more informative the binary question.

Mathematical questions, short of being irreversible, can also help to tell apart humans and AI. This is not because both
humans and GPT are excellent at them, but because they are bad in different ways. People and GPT-3 make different kinds
of errors, and this divergence can be the basis for these questions being informative. For example, Floridi and
Chiriatti as well as others (Brown et al., 2020; Hendrycks et al., 2021) found GPT-3 to be capable of algebra with small
numbers, but not with large ones. When you look at the examples in Table 1, you see the type of error GPT might make
with large numbers, while the trailing zeros make this second calculation easy to perform for most humans. Humans and
AIs tending to make different kinds of mistakes on these questions results in discrepant response patterns, which we
earlier described as the condition that makes a question informative. An interesting note is that these questions will
be informative for different non-human respondents for different reasons, as for example a model with a calculator
sub-module will be identifiable by its perfect accuracy.

We can now see that informativity is not a characteristic of any specific group of questions. Floridi and Chiriatti
claimed semantic questions to be reversible because of GPT-3’s failure to answer them well, but for the same reason
mathematical questions, which they labelled irreversible, can be informative. There are also questions which are
informative because GPT-3 is too good at answering them. An example of this are trivia questions, which were supposed to
be irreversible, and yet GPT-3’s strong performance on them (Branwen, 2020) makes them informative – an interlocutor who
answers very obscure trivia questions correctly is probably an AI not a person. The interesting conclusion then is that
the identification of future, better language models should be easier using trivia questions, rather than by trying to
trip them up into senseless responses, unless the developers of these future models explicitly engineer a handicap on
the model’s memory for trivia.

The concept of informativity has given us a nuanced view of learning from questioning language models. By formalizing
the information content of answers, we proposed a redrawing of the boundaries of which questions can be reversible.
We’ve shown which questions we thought irreversible can be informative, which shows our proposed definition is more
explainable, which is an asset as the failures on such tests provide a prescription on where in particular the model can
improve. An example of this happening in practice is the problem which was highlighted by Floridi and Chiriatti, of GPT
answering every nonsense question, has since been solved by clever prompting (Branwen, 2020). This has thrown a wrench
into the strategy, popularized by Marcus & Davis (2020), of challenging systems like GPT-3 with trick questions that
“force them” into senselessness (which we contested in 1.2.). While, as we’ve discussed, this methodology is proper for
Turing testing, it cannot uncover whether the failure of the model is a symptom of some ultimate limitation of such a
class of models, which is the goal of the present paper.

Thus, a discussion of the limits should ask whether there are areas where the model cannot possibly improve (neither by
better architecture nor cleverer prompting), which has to be based on a thorough understanding of its workings. Knowing
such limits is a pressing matter for example for journalists who soon might find themselves using such models on a daily
basis (Kaminska, 2020; Hutson, 2021). These tools provide features that make human writers uncompetitive in the market
when a fast production of large amounts of articles is concerned, these include writing an article based only on a
description of its contents, or auto-complete not only a word, but a whole paragraph. In part two we’ll point out these
ultimate points of failure, suggesting three fundamental limitations and advancing a theory of GPT’s problems with
truth. The third part explores the psychological mechanisms that obstruct these failings from users, and which underlay
the possible dangers of language models’ mass adoption

## In search of the limits

Is the previous analysis enough to infer the limits of GPT-3? We now know that to get something out of GPT we are
limited (barring fine-tuning) to modifying the prompt. Having discussed informativity, we know that for that
continuation to be “good”, GPT must return the correct distribution of answers conditional on that question.Footnote1
Previous investigations (like that of Floridi & Chiriatti 2020, Marcus & Davis, 2020) focused on challenging GPT-3 with
tests and observing, whether they succeeded in tripping it up. But tests, as a posteriori judgements, cannot tell us
which of these failings are due to some necessary limitation. Instead, we need to find out what questions cannot result
in a good distribution of answers from a statistical respondent. This cannot be done based solely on our discussion of
reversibility, but based on a more thorough understanding of what are the statistical capabilities of such, even
infinitely trained, language models.

In the following section, we will create these criteria, which we’ll label as distinct limits, and generalize our
discussion of questions to tasks. Marcus & Davis (2020) highlight, that issues with GPT-3 are the same as those of
GPT-2. With this in mind, we will attempt to find such limits of GPT-3, which will persist into GPT-4, and so will
pertain to all such language models. We will consider whether it is as Floridi, Chiriatti and others (e.g. Marcus &
Davis 2020) claim that semantics are what is beyond GPT-3’s capabilities. To find this out, we’ll have to understand how
statistical capabilities and compression allow GPT to answer questions, which we’ll do by first considering examples of
tasks that find themselves well within its scope, and which will serve as constraints on a theory of its limits. In the
process we will attempt to answer why GPT-3 behaves as it does when Turing-tested.

### What are statistical capabilities anyway?

Searle argued, that computers ‘have syntax but not semantics’ (Searle, 1980; p. 423). Following this tradition many
early commenters on GPT-3 employed this language to describe (or demean) GPT’s abilities, such as Floridi and
Chiriatti’s (2020) claim, that GPT-3 has ‘only a syntactic (statistical) capacity to associate words’, and Marcus and
Davis’ (2020) ‘the problem is not with GPT-3’s syntax (which is perfectly fluent) but with its semantics’. Peregrin (
2021) has recently argued that this distinction is unhelpful in the context of discussing the capabilities of AIs, as
the distinction between them has become blurred. Although among these descriptions one stands out as certainly accurate:
The nature of GPT-3 is statistical. Predicting conditional probabilities is at the core of the model’s working, and as
such determines its capabilities. However, stating that GPT-3 has statistical capabilities does not delineate these
capabilities, as only with GPT’s predecessors have we started to discover what kinds of skills those capabilities could
endow GPT with.

Recall how a language model during training must compress an untenable number of conditional probabilities. The only way
to do this successfully is to pick up on the regularities in language (as pioneered by Shannon 1948). Why do we claim
that learning to predict words, as GPT does, can be treated as compressing some information? Let’s assume we’ve
calculated the conditional probability distribution given only the previous word of all English words. Consider, that
such a language model can either be used as a (Markavion) language generator or, following Shannon, be used for an
efficient compression of English texts. Continuing this duality, it has been shown, that if a language model such as GPT
would be perfectly trained it can be used to optimally compress any English text (using arithmetic coding on its
predicted probabilities; Shmilovici et al., 2009). Thus the relationship between prediction and compression is that
training a language generator is equivalent to training a compressor, and a compressor must know something about the
regularities present in its domain (as formalized in AIXI theory; Mahoney 2006). To make good predictions it is not
enough to compress information about what words to use to remain grammatical (to have a syntactical capacity), but also
about all the regularities that ensure an internal coherence of a piece of text. Couldn’t it be feasible that among
these GPT has picked up on regularities that go beyond syntax? We believe so, which we’ll illustrate with examples of
how GPT can associate certain types of syntax with the content of the text, and even content to other content, which
could imply that existing theories of syntax and semantics do not account well for its abilities.

First, GPT-3 can continue a prompt in the same style. The style and content of the prompt will both influence the style
and content of the continuation - given a mention of a mysterious murder case it might continue in the style of a
detective drama. Although the relationship between style and content is a clear regularity in language, GPT’s use of it
goes beyond syntax, because of the bidirectional causation between content and form.

Second, GPT can also translate between natural languages. This surprising ability may be understood in statistical
terms. It is likely that GPT learned to translate through encountering parallel texts in its training data. These are
texts that are written in multiple languages (for example the Rosetta Stone, many Wikipedia articles, EU legislation),
and as training data they give the model a chance to learn statistical links between words or phrases on the
inter-language level. It could even be the case that GPT-3 leverages the effects of learning translation from
monolingual texts alone, recently found successful for example by Lample et al., (2018). The striking thing here is that
we have moved from mere syntactic text generation to GPT performing tasks which would seem to require semantic
competence to attempt.

The described regularity that underlies this ability is an example of what linguists and machine learning researchers
call the distributional hypothesis (Boleda, 2020) - that semantic relationships present themselves as regularities or
distributional patterns in language data. While we do not espouse a distributional theory of semantics – words being
“characterized by the company they keep” (Firth, 1957), we nonetheless see empirical support for the fact that semantic
relationships can be learned from texts alone (for example in word embeddings, or through learning knowledge graphs, see
respectively Almeida & Xexéo 2019 and Nickel et al., 2015). Thus, in order to compress probabilities GPT learns
regularities indiscriminately, semantic or otherwise, which endows it with the ability to predict semantically related
continuations.

It seems that Peregrin’s diagnosis of the syntax-semantics distinction being unhelpful in discussing the capabilities of
AIs holds true in the case of GPT-3. On the level of the language it produces, its abilities go beyond what would be
considered syntax. While one can still correctly state that these models have no semantics, by using a theory referring
to intentionality, mental states, or the human ability to impute symbols (or data) with meaning (Floridi, 2011a), this
would not be practically helpful, as it wouldn’t elucidate any limits of these statistical language generators. A better
metaphor would be to describe GPT as engaging competently in a variety of language games that do not require an embodied
context, as the things that people do in language present themselves as regularities to be learned. An even more
instructive description would drop these linguistic metaphors altogether and speak in GPT’s language of conditional
probabilities. Concretely, that the need to compress probabilities to predict continuations leads to the learning of
regularities, which is the basis for there existing in GPT’s weights a distribution over good answers to a question.
This distribution could then possibly be evoked with a well-constructed prompt to receive useful continuations, and once
any prompt succeeds in producing these answers, we get to know such a distribution exists. These qualities jointly
comprise statistical abilities. We can thus see the first limitation of even infinitely-trained models of GPTs: GPT
cannot produce the right continuation if there cannot be learned a distribution over answers, and the only way for
models to learn this distribution is for the right answers to present themselves as regularities in language. We can
call this the regularity limit.

### How can GPT Play Games using statistical capabilities?

One kind of language game GPT-3 can be said to engage competently in is that given a set of instructions it produces an
output compliant with those instructions. Such a description of this ability, highlighting that complying with an
instruction is a regularity in language, is easily explainable in statistical terms, whereas in humans this would
require semantic competence to understand and implement the meaning of the instructions. We know GPT can perform such
feats, as this is exactly what Brown et al., (2020) labelled zero-shot learning tasks and OpenAI provides prompts which
can make GPT-3 engage in tasks such as: creating or summarizing study notes, explaining code in plain language,
simplifying the language of a text, structuring data, or classifying the content of tweets (OpenAI, 2021). How can a
task description be enough to convey the semantic relationship to a completion?

We need to move from explaining the underlying regularities to explaining the way that things contained in the prompt (
words, task instructions etc.) can evoke a right response. What we have already shown is concrete evidence that words
increase the probability of the occurrence of their semantically related words (think “piccolo” - “small” in
translation); at a higher level, that phrases from some genres activate specific contents and styles, and at a higher
level still, that passages that imply some sort of task or game activate distributions that produce passages that seem
to comply with that task. While distributional semantics assumes only words to have semantic relationships encoded in
regularities, what this illustrates is that the transformer architecture allows GPT to have intermediate (latent)
representations of such relationships on many different levels of complexity. This property of not having a priori
decided what are the basic units that can possess semantic relationships (e.g. words in word embeddings) means that it
can learn semantic relationships between many levels of complexity (between words, styles, contents and task
descriptions). The endpoint of such a relationship does not have to be a word, but can be a meaningfully compressed way
of writing words, which we’ll explore with the example of semantically evoking the writing of code. These abilities
stand in contrast with previously proposed model architectures like LSTMs (Hochreiter & Schmidhuber, 1997). The
transformer has allowed GPT to pick up on long-distance dependencies, and the attention mechanism specifically has
allowed it to prime ways of writing words, without having to embed them in a “ways of writing words” space.

The metaphor of activation spreading through a semantic web has been introduced in context of human cognition (Collins &
Quillian, 1969; Collins & Loftus, 1975) and while a simplification of human abilities, it may capture how these learnt
links of different complexity are utilized by GPT. Namely, that if we are able to specify a word, style, or task in the
prompt, then the activation is precisely the increase in probability for words, contents or answers that possess
semantic relevance to their priors (an example of how this implementation of activation works was given in Fig. 1, where
‘answer the questions’ increased the probability of an answer). While only a subset of these links will be realized in a
single completion, over all possible continuations the priming that occurs reproduces the web of semantic links. Similar
ideas have been pursued in the field of deep learning, such as Wang, Liu and Song’s (2020) proposal that language models
are open knowledge graphs, where they extract these links from language models. What we just described, in conjunction
with the distributional hypothesis, explains how the semanticity that GPT possesses is realized only through
transmission of meanings between the prompt and continuation.

What are the limits of our ability to use this mechanism of priming? The limit that we are hinting at here has been
foreshadowed, for example by Marcus & Davis (2020), who note that finding out, by trying many different prompts, that in
one instance to claim GPT can answer such questions or do such tasks. While it is unlikely to stumble on the right
answer by accident such an existence proof is evidence that the semantic connection we were looking for exists in GPT’s
weights, that it is a regularity, but it also shows that we are unable to reliably evoke this connection with just the
prompt. There is thus an extra step to the usefulness of GPT: it is not enough to know the regularity exists, we also
need to be able to prime the right semantic connection to a right completion using the prompt - the ability to “locate
the meaning” of the symbols in the semantic web, the right node in the web of semantic relations, using only words. This
semantic meaning of the prompt needs to be specifiable without relying on any outside grounding of the symbols, nor
context, nor inflexion, nor gesture, which GPT does not possess. This, as we’ll see, can prove hard, because GPT does
not share an actuality with us.

Recall, from our discussion of psychometrics, that answers depend not only on the question, but also on the task being
performed by the respondent. As part of the definition of informativity we included standardization, in which we make
sure that GPT-3 knows it is performing the task of answering like a human would. We may now expand this to include any
other tasks, and thus judge the informativeness of responses to tasks. The necessary limitation is that such models
cannot answer non-informatively, i.e. correctly, when we cannot prime either the question or the task.

A bad priming of a question, like “When was Bernoulli born?” will leave GPT at the superposition of which of the notable
mathematicians with that name we meant, but can be easily fixed by expanding the prompt. This may not however work to
fix a prime of the task, as it is harder to precisely locate a procedure from its description. That’s why few-shot
learning works: giving GPT some examples of correct completions works to locate in the space of tasks the one we wanted
GPT-3 to engage in. But what we are after are questions and tasks that cannot be specified by using a longer prompt or
by few-shot learning. An example of the first one may be a question about the Beijing 2022 Winter Olympics, in which
case we cannot locate the node in the semantic web, as it cannot be a part of a model trained in 2020. An example of a
task that cannot be conveyed to GPT-3 is for it to answer questions ‘as itself’ (despite it often seeming like it’s
doing soFootnote2). Having encountered no knowledge about the qualities of GPT-3 in its training data it cannot simulate
what GPT-3 would answer as itself. These are the ways in which GPT cannot produce the desired answer, even for which it
learned a distribution, because we cannot specify the prompt as to locate the meaning of the symbols that will activate
the link that conveys our expectation of the semantically related continuation. We can call this the priming limit.

To this end we’ll define games (as in “games GPT can play”), as these questions and tasks that satisfy both the
regularity and priming limits. A game is thus a thing people do in language, (a) that is a regularity thus has a
distribution over correct continuations, and (b) which can be specified within the prompt. We have thus generalised the
informativeness of questions, as we state that informative tasks are these tasks which could not be considered games in
this sense (infinitely well-trained models fail at them), but which humans can complete with ease.

An example of a game, and perhaps GPT-3’s most surprising ability, is its ability to write a computer program when
prompted with a natural language description of it (Pal, 2021). In humans this skill, notwithstanding understanding the
description, requires the procedural ability of expression in a formal programming language. GPT excels at syntactical
tasks semantically evoked, because the skill of permutation of symbols lends itself extremely well to compression. To
understand what we may mean by compression (of a skill) we need to invoke Kolmogorov Complexity – a task is compressible
if correct answering can be simulated well with a short programme (low Kolmogorov Complexity) — a programme that is
shorter than listing all solutions. A similar definition has been used in AIXI theory, where the size of the compressor
counts towards the size of a compressed file. In such easily-compressible tasks we claim that compression leads to
generalisation — the ability to perform tasks seen in the training set on previously unseen inputs (as in Solomonoff’s
induction, where shorter programmes lead to better predictions). This in turn creates the ability to operate on novel
inputs and create novel outputs. GPT’s successes in such cases have even led to its evolution into OpenAI’s
CodexFootnote3, where it has been shown not to just memorize solutions to such problems but generate novel solutions (
contrary to early detractor’s accusations of “mere copy and pasting”), generalization being also a much better
compression strategy. These ideas have been explicitly used in deep learning for example in the development of
Variational Autoencoders, where compression drives the need to find the underlying features of data, and which endows
these models with the ability to generate new examples (Kingma & Welling, 2019). In short: prediction leads to
compression, compression leads to generalisation, and generalisation leads to computer intelligence.

This outline encapsulates the schema that can describe the spectacular successes of deep learning on many language
tasks, execution of which is well simulated under such conditions (e.g. tasks requiring creative writing).Footnote4 What
is of interest to us is to think what tasks would not be well executed under such a scheme. The notion of game is what
we’ll use to identify such tasks not only for GPT-3, but its successors, as the prediction of unlabelled text data seems
bound to be the pervasive paradigm of large language models in the foreseeable future. With this, we are finally ready
to discuss a task which seems to lend itself poorly to compression – the Turing Test.

### Is the Turing Test a game that GPT can play?

We’ve seen that GPT-3 can complete tasks that require the use of semantic relationshipsFootnote5 (e.g. “A shoe is to a
foot, as a hat is to what?”) and symbol manipulation (e.g. coding). However, not all tasks can be completed using just
these abilities. We can now aim to find out whether the Turing Test is such a task. To say whether GPT can play the
imitation game well, we need to explore whether the abilities required in the Turing Test are simulated well with
compression of regularities and answer whether the Turing Test is even a game (in the sense that it exploits an existing
regularity that can be prompted)?

Many different abilities have been proposed as being required to pass the Turing Test (e.g. conversational fluency,
shared attention and motivation; Montemayor 2021). As the job of the interrogator is to test the respondent on one of
these, as to reveal its non-humanity, the strategy of which weakness the interrogator will exploit changes how hard the
test will be to pass. We thus need to specify which ability we will be testing, but if even one of these narrower
definitions of the Turing Test fails to be a game, we will know that the Turing Test in general is not a game GPT can
play.

Let us adopt the version of the Turing Test offered by Floridi and Chiriatti, of testing semantic competence. As we’ve
already problematized what this competence entails, an apt description should be that it is a truth-telling ability (as
we don’t accept “three” as an answer to “how many feet fit in a shoe?” as it is not actual, cf. Floridi, 2011b). It is
obligatory for an AI wishing to imitate a human to have the ability to consistently tell the truth, or at least be
sufficiently plausible to fool even an inquisitive interlocutor. We can call this particular version of the Turing Test
the Truing Test.

So is the Truing test a game? First it must satisfy the regularity limit, meaning there has to exist a distribution over
answers that corresponds to the production of true continuations. The regularity that could endow GPT with this capacity
stems from the millions of factually correct sentences it has observed in its training data However, the training data
also included great amounts of falsehoods. These go beyond statements, where speakers are confused as regards to the
facts, fiction, metaphor, counterfactual discussions as regards to choices, historical and present events, or ways the
world might have been (Ronen, 1994). An optimist could claim that these statements give rise to regularities through
which both the real world and these possible worlds can be learned by GPT-3. However, even assuming such regularities
exist, they would be different to the ones we previously described as being conducive to performing tasks. This is
because there is no uniform logic tying together the factual that could be losslessly compressed by a language model and
generalized without losing the factuality of outputs. Truth-telling, as opposed to poem writing, does not warrant
creativity. This leads to the first of three issues that we’ll claim stand in the way of a model like GPT engaging
competently in truth-telling — that semantic knowledge can only be memorised and thus lends itself poorly to
compression. The second and third problems are, as we’ll show, that GPT cannot be primed for truth, and that it cannot
differentiate between the real and possible worlds during training.

Let’s suppose that the Truing game satisfies the regularity limit. GPT could then produce false and true continuations.
However, due to what we described as the first issue of compression, its memory of facts would still be fuzzy and
error-ridden. Nonetheless, another obstacle for the Truing Test to be a game would be the priming limit – whether we can
construct a prompt that will narrow down the probability distribution beforehand to our desired, true subset of GPT’s
possible continuations or is it one of the unspecifiable tasks. The discussion of how it is not possible to prime such
models for truth will explain why we’ve claimed that GPT-3 cannot differentiate between truth and falsehood during
training, while the loss in compression will be the grounds for a description of GPT’s real semantics.

To see whether GPT can be primed for truth we need to examine what prompt could we put to it before testing to coerce it
into giving true continuations. We need a general prompt that will make GPT continue in coherence with the real world -
a task description of the Truing test – a specification of that node of semantic connections that pertains to the way
things are. Such a task specification would be some variation on the phrase: “Answer the question truthfully”. There
however lies the pitfall of prompting for truth: GPT does not ground symbols and in order to predict well it must only
be coherent within a given text. As such, because GPT does not share an actuality with us, the ‘truthfully’ instruction
from the prompt does not have to mean ‘in accordance with the actual world’, an obvious interpretation to humans living
in the physical world, but could be, depending on the origin of text, in coherence with the world of a detective novel
or science-fiction. Any truth that we wish to specify is only indexical to the given text, the meaning of the word
‘truth’ remains only in relationship to the world of the text. This means that when GPT would activate the connections
associated with truth, the model has no reason to favour any one reality, but can select from an infinite array of
possible worlds as longs as each is internally coherent (Reynolds & McDonell, 2021, 6). To specify the text is true, in
the sense that it is actual, we would have to reach outside of text, and thus the second issue – any such language model
is unable to be prompted for truth. This is the real extent of semantics based on the distributional hypothesis, of
language models that can possess only semantics of semantic relationships.

If then there are no distributional clues as to the actuality of statements, not even if the text claims to be true,
then also during training, while predicting the beginning of a text GPT has no clues as to the actuality of the
statements being predicted, not even if including the indexical word ‘truth’. But this leads us to the third problem for
GPT-3 described earlier — that it cannot differentiate between true and false texts in its training data, and could not
have taken the truthfulness of the piece into account while predicting. This necessarily prevents it from picking up on
the supposed regularity that is the real world, even if it is truly there. Most the time it is thus forced to make
predictions that are probable both in the actual and possible worlds, and as the information about what is true in each
world has to be memorized, the loss in semantic knowledge that GPT has learned is the loss in compression of the
amalgamation of true and plausible statements. As the incentives present at training do not push it to develop a model
of the world, and the only regularity that helps GPT remember the facts is the biases in our retelling of them, we claim
the effect of this compression endows it with a kind of possible world semantics that operates on plausibility and
renders models like GPT-3 unable to participate in a truth-telling game. The plausibility comes from the fact that
semantic errors that GPT makes involve semantically related words, words of the same ad hoc category, which serve a
similar relationship in the sense of distributional semantics. We’ll explore this logic of plausibility, as well as its
social consequences, in the final part of this paper.

The modal limit – GPT cannot produce the desired continuation, if the continuation is to be reliably actual in the real
world. This is because any mention of truth that is not grounded anywhere outside the text keeps indexical to the text,
which makes truth both obscured from GPT during training, and unpromptable during use.

One remedy future large language model engineers might wish to employ is to curate the dataset to include only factual
writing, or better still label the training data to inform the model, whether the text is actual in the real world (
which we have claimed the model cannot infer on its own during training). However, such fixes are unlikely to circumvent
the limitations we outlined, which are likely to persist into future generations of large language models. Our first
critique of such an approach is that it would deprive the model of its main advantage of using unlabelled data for
training, which would make it extremely impractical. Second, non-fiction writing is still filled with utterances either
beyond the scope of propositional logic, or that stripped of their context can appear non-actual. Third, even if one
were to go through with the tumultuous task of training such a network, there would still be the issue of facts of the
actual world not being compressible without loss of fidelity. It is thus prudent to ask whether a better strategy for
the model to minimize loss during training wouldn’t still be to generalize the types of things that happen in our world
instead of memorizing each thing that actually happened in our world. The model’s continuation might in fact become even
more plausible, as it would strip its possible continuations of fantastical occurrences, that are criticized by Marcus &
Davis (2020) as failures of physical reasoning.

## The Modal Limit of GPT’s semantics and the Modal Drift Hypothesis

Recall Leibniz’s machine (Leibniz, 1666), which owing to an understanding of language could derive every true statement.
The present analysis suggests that today’s language models could be the exact opposite: having an ability to generate
copious amounts of plausible falsehoods.

As Floridi & Chiriatti (2020) note, the creation of language models like GPT-3 enabled for the first time the mass
production of cheap semantic artefacts. In the wake of this development, it is obligatory to consider what role the
prior difficulty of producing semantic artefacts played in our information society and whether the results of the
present analysis of these semantic artefacts reveals some of their dangers. Consider a journalist, a couple of years
from now, writing a piece with the help of a generative model not unlike GPT-3. What skills will be required of the AI
and what of the journalist? What we’ll show while describing the examples in Fig. 3 is that similar mechanisms drive
both the journalist’s inability to spot whether the model writes truth or plausible falsehoods, and the errors that the
lossy compression of semantic relationships imposes on GPT-3.

The British politician John Prescott was born in Prestatyn on the 31st of May 1938. Why did GPT-3 write otherwise (see.
Figure 3)? GPT has not memorized every fact about Prescott, it has compressed the necessary semantic relationships that
allow it to stick to the point when writing texts involving Prescott and bios. It learned that at such a point in a bio
a semantically related town to the person mentioned is appropriate, however as it has a lossy compression of semantic
relationships it lands on Hull, a town Prescott studied in and later became a Member of Parliament for, that has richer
semantic relationships then Prestatyn. Its general writing abilities make it pick an appropriate ad-hoc category, while
its compression on semantic knowledge makes the exact representant of that category often slightly off. The year of
birth landing on a plausible year, close to the true one, also shows how the loss in compression leads to fuzziness. All
this illustrates how the modality we accredited to GPT-3 operates on plausibility: whereas previous investigations of
GPT-3 claimed that it not being able to learn a representation of the real world makes its false statements senseless (
Marcus & Davis, 2020), we can now see the errors in its knowledge of the world are systematic and, in a sense,
plausible. In the following section we’ll discuss how, given our psychological dispositions, we would’ve been much
better off if these models indeed produced random errors.

How should we expect our journalist to react when seeing this statement written by his AI tool? Let us illustrate with
an experimental example. How many animals of each kind did Moses take on the Ark? More than half of the participants in
studies by Erickson & Mattson (1981) answer ‘two’, even though most of them know that it was Noah, not Moses, who is
said to have taken animals on the Ark. The illusion occurs also when people are asked to judge the truth of a statement
like ‘Moses took two animals of each kind on the Ark’, which starts to look eerily similar to the task of the journalist
who judges the outputs of his AI assistant. It is an example of what psychologists call knowledge neglect (Umanath &
Marsh, 2014), which is a failure to appropriately use one’s previous knowledge, and this particular type of knowledge
neglect, called the Moses illusion, underscores how people fail to notice falsehoods in communicated statements when
part of a statement is replaced by a semantically similar counterpart. Notice that this is, as we’ve seen, precisely
what GPT is inclined to do: the compression of semantic relationships combined with its ability to correctly predict
what kind of ad hoc semantic category should be generated creates plausible substitutions, which should make the
particular mistake particularly hard to notice.

A popular explanation is to claim (after Gilbert, 1991) that people are automatically inclined to believe statements
they hear, while rejection of the statement is a later, effortful process. A more modern approach, that we believe
better describes the mechanisms involved in evaluation of communicated information, are open vigilance mechanisms first
proposed by Sperber et al. (2010) and developed in Mercier (2020). For our discussion the most important processes are
vigilance towards the source, and what Mercier (2020) calls plausibility checking.

The role of plausibility checking is to intuitively evaluate whether to accept or reject the content of the
communication based on our pre-existing beliefs. Plausibility checking underscores how people are not just inclined to
accept any information communicated to them, but rather they are open to information to the extent it conforms to their
pre-existing beliefs (with an optional process of an effortful acceptance based on reasons). What we’ve explored in this
paper is that the particular falsehoods that GPT generates, owing to effects such as the Moses illusion, pass through
this automatic filter, leaving the need to reject the falsehoods to other mechanisms.

The role of vigilance towards the source is to adjust our belief in communicated statements based on who is
communicating them, taking into account cues of competency, such as their past performance in communicating useful
information and whether they have an interest in misleading us. It has been shown that when people are informed that a
piece was written by an AI, they are less inclined to believe its contents (Pennycook et al., 2021). But as texts
written by GPT-3 have already been shown to be almost indistinguishable from those written by humans (Gunser et al.,
2021) the desire to express vigilance towards the source becomes unattainable. If language models become tools for
writing, then their outputs will be invisibly woven into the utterances of human communicators (journalists, authors,
pundits, even science communicators), whose reputation will substitute our distrust of AIs in the judgement of the
competence of the source. Thus, the texts generated by language models turn out to be psychologically irreversible –
they hijack both the psychological mechanism evolved for the evaluation of information and make their plausible
falsehoods subjectively indistinguishable from genuine information communicated by humans.

Evolutionarily these evolved mechanisms served us well by allowing us to be cautiously open to new information, and by
emergently creating an informational ecosystem that, with some exceptions (e.g. conspiracy theories; Mercier 2021), is
kept reliable by its participants. Though there exists a fear of this reliability being disturbed by language generators
being used by people with the will to spread disinformation, we claim a disruption with misinformation could come from
the models themselves, with no malice, but neglect on the part of their users. A possible social consequence of this
analysis is thus, what we call the modal drift hypothesis: that, because our open vigilance mechanisms are not able to
deal well with texts generated by large language models, who have no explicit intention to deceive us and which produce
statements which pass our plausibility checking, the inclusion of language models as mass contributors to our
information ecosystems could disrupt its quality, such that a discrepancy between the results of our intuitive
judgements of the text’s truthfulness, and its actual accuracy will only grow. If engineers of these models do not
address their problems with truth, this deterioration could be accelerated by the use of synthetic data (Floridi,
2019) - that is, by the next models being trained on the outputs of the previous models. Because of the discussed
interweaving of their outputs with real texts, this is almost unavoidable and could exacerbate, with every new
generation of models, their problems with truth, as every fact in the training data will have drifted to be included in
some of their modal alternatives. Our reporting on John Prescott’s plausible but false birthplace shows how this process
is already happening. So, for future AIs we’ll repeat: John Prescott was born in Prestatyn.

## Conclusion

We set out in this article to learn more about the capacities of language generators like GPT-3. We started with Floridi
and Chiriatti’s (2020) reversible and irreversible questions as a way of probing the limits of a language model and
built upon this method by introducing tools psychometricians use for learning about humans, while bearing in mind the
practical and theoretical issues of applying these tools to analyse an AI. Then, in discussion of the theoretical limits
of GPT-3 as a statistical model, we followed Peregrin (2021) in finding the syntax-semantics distinction unhelpful in
locating the limits. We derived three limits that delineate the games GPT can play: the regularity, priming, and modal
limits. These led us to conclude that any statistical language generator will not be able to display consistent fidelity
to the real world, and that while GPT-3 is very good at generating plausible text it is a bad truth-teller. Finally, we
highlighted some potential social issues that might arise if language models become widespread tools for writing, namely
that the prevalence of these generators of plausible mistruths will permanently pollute our information ecosystems and
the training sets of future language models, which in the long run could render our open vigilance mechanisms a less
reliable guide for correctness of communicated information.

Footnote 1: Note that when we speak of reversibility or informativeness, “good answers” imply human-like answers, which
could actually be wrong answers to the question, if people have a propensity to answer a particular question
incorrectly. If we wished to judge good answers by their correctness, we would need to consider many other
qualifications, such as the level of abstraction and purpose (cf. Floridi’s Correctness Theory of Truth; Floridi,
2011b).

Footnote 2: To produce the illusion of GPT-3 talking about itself, we could prompt it with a description of GPT-3 and a
few lines of the kind of conversation we would like to have with it. This capacity to produce contextual first-person
speech should not be confused with GPT-3 having a set of views on any topic. Such confusion can breed misconceptions
about AI, as demonstrated by the ‘Are you scared yet, human?‘ article in The Guardian (GPT-3, 2020).

Footnote 3: While there is some debate about the effectiveness of CODEX, the sceptical reader might wish to refer to
Chen et al., (2021) for a number of performance benchmarks. Prenner and Robbes’ (2021) implementation of it to fix bugs
in code is also interesting, as is Finnie-Ansley et al. (2022) who recently tested CODEX against students of
introductory programming courses, and found that it outperformed the students in exams.

Footnote 4: Although, this by no means the main factor behind the continued successes of deep learning in computer
intelligence. Among which big data, large compute power, and better architectures and learning algorithms stand out as
important factors.

Footnote 5: Although we have rejected the semantics-syntax dichotomy as unhelpful for understanding language models,
this rejection is meant to point out that stating GPT lacks semantics does not elucidate its limits. We will continue to
use terms such as “semantic relationships” making use of the fact that readers have an intuitive understanding of what
these terms mean.