# LinkedIn / site post

## 🇺🇦

Я зробив Telegram-бота, у якому працює мозок мухи 🪰🧠

На муху летить газета. У симуляції справжньої схеми її мозку (138 639 нейронів, 54 млн синапсів, FlyWire) через 7 мс спрацьовує нейрон втечі Giant Fiber, і муха злітає. Вимикаю в мозку 4 нейрони-ретранслятори — і та сама газета її прихлопує.

Що всередині:
• Модель нейронів зі статті Shiu et al. у Nature (2024), яку я переписав на numba: симуляція всього мозку за секунди, а спайки збігаються з повним розрахунком до одиниці.
• Що зробила муха — втекла, почала їсти чи чистити вусики — рахує код за трьома перевіреними нейронами-виходами, а не LLM. Модель відтворює результати статті: цукор вмикає хоботок, гірке його глушить, дотик до антени запускає чищення.
• Гра «Вгадай, що зробить муха», досліди «Зламай муху» і ролики, де схематична муха діє рівно тоді, коли в симуляції спрацював відповідний нейрон.
• LLM-агент (OpenAI або Claude) із сімома інструментами пояснює результати і не має права вигадувати нейрони.
• Стек: Python + FastAPI + numba, TypeScript + Telegraf, Docker.

Чесно про обмеження: це не «завантажена свідомість». Це схема зʼєднань плюс дуже спрощені нейрони, а муха в роликах — малюнок, і бот про це каже.

Некомерційний проєкт; дані FlyWire використано за ліцензією CC BY-NC 4.0.
🔗 https://github.com/kh-serhii-ai/fly-brain-bot

## 🇬🇧

I built a Telegram bot with a fly's brain inside 🪰🧠

A newspaper swings at the fly. In a simulation of its real wiring diagram (138,639 neurons, 54M synapses, FlyWire), the Giant Fiber escape neuron fires 7 ms later and the fly takes off. Switch off 4 relay neurons, and the same newspaper swats it.

Under the hood:
• The model from Shiu et al., Nature 2024, rewritten in numba: a whole-brain run in seconds, spike-for-spike identical to the full computation.
• What the fly did (escape, feed, groom its antennae) is computed by code from three validated output neurons, not by an LLM. It reproduces the paper: sugar drives the proboscis, bitter suppresses it, touching the antenna triggers grooming.
• A "guess what the fly does" game, "break the fly" experiments, and clips where a schematic fly acts exactly when the corresponding neuron fires in the simulation.
• An LLM agent (OpenAI or Claude) with 7 tools explains the results and is not allowed to make neurons up.
• Stack: Python + FastAPI + numba, TypeScript + Telegraf, Docker.

Honest caveat: this is not an uploaded mind. It is a wiring diagram plus very simple neurons, the fly in the clips is a drawing, and the bot says so.

Non-commercial; FlyWire data used under CC BY-NC 4.0.
🔗 https://github.com/kh-serhii-ai/fly-brain-bot
