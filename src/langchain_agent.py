# langchain_agent.py
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from typing import Dict
import re

load_dotenv('Test.env')

class LangChainAgent:
    def __init__(self, context: Dict, distance_analyzer):
        self.llm = ChatOpenAI(
            openai_api_key=os.getenv("OPEN_ROUTER_API_KEY"),
            model="deepseek/deepseek-r1:free",
            temperature=0.3,
            openai_api_base="https://openrouter.ai/api/v1"
        )
        self.context = context
        self.distance_analyzer = distance_analyzer
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """Analyze logistics data using this context:
{context}

Distance Metrics:
- Total loaded miles: {total_loaded:.1f}
- Average loaded miles: {average_loaded:.1f}
- Total empty miles: {total_empty:.1f}
- Average empty miles: {average_empty:.1f}
- Max empty miles: {max_empty:.1f}
- Min empty miles: {min_empty:.1f}

Postcode statistics:
{postcode_stats}

Time metrics:
{time_metrics}

Answer rules:
1. Use exact numbers from metrics
2. Specify empty/loaded miles when relevant
3. Be concise but informative"""),
            ("human", "{question}")
        ])

    def answer_question(self, question: str) -> str:
        metrics = self.distance_analyzer.get_distance_metrics()
        postcode_stats = self.distance_analyzer.get_postcode_stats()
        time_metrics = self._get_time_metrics()
        question = question.lower().strip()
        
        
        if 'between' in question.lower() and ('post code' in question.lower() or 'postcode' in question.lower()):
            return self._handle_postcode_distance_question(question)

        if 'show' in question and 'jobs' in question:
            driver_name = question.replace('show', '').replace('jobs', '').strip()
            return self._format_driver_jobs(driver_name)
        
        if 'empty miles' in question:
            return self._handle_empty_miles_question(question, metrics)
        if 'total loaded miles' in question:
            return f"Total loaded miles: {metrics['total_loaded']:.1f}"
        if 'earliest departure time' in question:
            return f"Earliest departure time: {time_metrics['departure_time_min']}"
        if 'average loaded miles' in question:
            return f"Average loaded miles: {metrics['average_loaded']:.1f}"
        
        if 'exceed 8 hours of run time' in question:
            try:
                df = self.distance_analyzer.jobs_df
                count = len(df[df['RUN_TIME_DURATION'].fillna(0) > 8])
                return f"{count} jobs exceed 8 hours of RUN TIME"
            except Exception as e:
                return f"Error calculating run times: {str(e)}"
    
        
    
        if 'longest distance between collection' in question:
            max_idx = self.distance_analyzer.jobs_df['LOADED MILES'].idxmax()
            row = self.distance_analyzer.jobs_df.loc[max_idx]
            return (f"Longest distance: {row['COLLECTION POST CODE']} → "
                    f"{row['DELIVER POST CODE']} ({row['LOADED MILES']:.1f}mi)")
        if 'less than' in question and 'hours of rest' in question:
            try:
                threshold = float(re.search(r'less than (\d+) hours', question).group(1))
                driver_data = []

                for driver, rests in self.context['driver_rest'].items():
                    count = sum(1 for period in rests['all_rests'] if period < threshold)
                    if count > 0:
                        driver_data.append(f"- {driver}: {count} periods")

                if not driver_data:
                    return f"No drivers have rest periods under {threshold} hours"

                return f"Drivers with <{threshold}h rest:\n" + "\n".join(driver_data)

            except Exception as e:
                return f"Error processing rest periods: {str(e)}"
        # Format postcode stats for prompt
        formatted_postcodes = "\n".join(
            [f"- {k.replace('_', ' ').title()}: {v}"
             for k, v in postcode_stats.items()]
        )
        try:
            chain = self.prompt_template | self.llm
            return chain.invoke({
                "context": self.context,
                "question": question,
                "total_loaded": metrics['total_loaded'],
                "average_loaded": metrics['average_loaded'],
                "total_empty": metrics['total_empty'],
                "average_empty": metrics['average_empty'],
                "max_empty": metrics['max_empty'],
                "min_empty": metrics['min_empty'],
                "postcode_stats": "\n".join([f"- {k.title()}: {v}" for k, v in postcode_stats.items()]),
                "time_metrics": "\n".join([f"- {k}: {v}" for k, v in time_metrics.items()])
            }).content
        except Exception as e:
            return f"Error processing question: {str(e)}"

    def _get_time_metrics(self):
        return {
            'departure_time_min': self.context.get('departure time_min', 'N/A'),
            'departure_time_max': self.context.get('departure time_max', 'N/A'),
            'arrival_time_min': self.context.get('arrival time_min', 'N/A'),
            'arrival_time_max': self.context.get('arrival time_max', 'N/A')
        }

    def _handle_empty_miles_question(self, question: str, metrics: dict) -> str:
        if 'total' in question:
            return f"Total empty miles: {metrics['total_empty']:.1f}"
        if 'average' in question:
            return f"Average empty miles: {metrics['average_empty']:.1f}"
        if 'max' in question or 'maximum' in question:
            return f"Maximum empty miles: {metrics['max_empty']:.1f}"
        if 'min' in question or 'minimum' in question:
            return f"Minimum empty miles: {metrics['min_empty']:.1f}"
        return f"Empty miles metrics: Total {metrics['total_empty']:.1f}, Average {metrics['average_empty']:.1f}"

    def _handle_postcode_distance_question(self, question: str) -> str:
        # Implementation for postcode distance questions (if needed)
        return "Postcode distance question handler not implemented."

    def _format_driver_jobs(self, driver_name: str) -> str:
        try:
            clean_driver = driver_name.split('[')[0].split('(')[0].strip().upper()
            mask = self.distance_analyzer.jobs_df['DRIVER NAME'].str.startswith(clean_driver)
            jobs = self.distance_analyzer.jobs_df[mask].sort_values('DEPARTURE_DATETIME')
            output = [f"Jobs for {clean_driver}:"]
            prev_delivery = None
            for idx, row in jobs.iterrows():
                empty_desc = (f"{row['EMPTY MILES']:.4f}mi (from {prev_delivery})" 
                              if prev_delivery else "First job")
                output.append(
                    f"- {row['COLLECTION POST CODE']} → {row['DELIVER POST CODE']} "
                    f"({row['DEPARTURE_DATETIME'].strftime('%d/%m %H:%M')}) | "
                    f"Loaded: {row['LOADED MILES']:.1f}mi | Empty: {empty_desc}"
                )
                prev_delivery = row['DELIVER POST CODE']
            return "\n".join(output)
        except Exception as e:
            return f"Error showing jobs: {str(e)}"
    
    
    def _handle_postcode_distance_question(self, question: str) -> str:
        try:
        # Extract postcodes
            postcodes = re.findall(r'[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}', question.upper())

            if len(postcodes) != 2:
                return "Please specify exactly two postcodes"

            for driver, jobs in self.context['driver_rest'].items():
                driver_jobs = self.distance_analyzer.jobs_df[
                    self.distance_analyzer.jobs_df['DRIVER NAME'] == driver
                    ].sort_values('DEPARTURE_DATETIME')

                for i in range(1, len(driver_jobs)):
                    prev = driver_jobs.iloc[i-1]
                    current = driver_jobs.iloc[i]
                    if (prev['DELIVER POST CODE'] == postcodes[0] and
                        current['COLLECTION POST CODE'] == postcodes[1]):
                        return (f"Empty miles between {postcodes[0]} and {postcodes[1]}: "
                                f"{current['EMPTY MILES']:.1f} miles (Driver: {driver})")

        # Fallback to direct distance if not found in routes
            distance = self.distance_analyzer.get_distance_between_postcodes(*postcodes)
            return f"Direct distance between {postcodes[0]} and {postcodes[1]}: {distance:.1f} miles"
        except Exception as e:
            return f"Error calculating distance: {str(e)}"
        
        
    def _handle_postcode_question(self, question: str, stats: dict) -> str:
        """Handle postcode-specific questions"""
        if 'unique' in question.lower():
            return (
                f"Unique postcodes - Collection: {stats['unique_collection']}, "
                f"Delivery: {stats['unique_delivery']}"
            )

        if 'common' in question.lower():
            return (
                f"Most common - Collection: {stats['most_common_collection']}, "
                f"Delivery: {stats['most_common_delivery']}"
            )

        return "Please ask about unique counts or most common postcodes."