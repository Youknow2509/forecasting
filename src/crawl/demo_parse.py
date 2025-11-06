import json
from src.crawl.table_parser import parse_html_to_records

HTML_SNIPPET = '''<tbody>
                                    <tr class="tralter">
                                        <td colspan="11">
                                            <strong>Tây Nguyên</strong>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td class="">
                                            <b>Đồng Nai 3</b><br /><small
                                                >Đồng bộ lúc: 17:30 06/11</small
                                            >
                                        </td>
                                        <td class="tdclass">06/11 12:00</td>
                                        <td class="tdclass">589.79</td>
                                        <td class="tdclass">590</td>
                                        <td class="tdclass">570</td>
                                        <td class="tdclass">271.26</td>
                                        <td class="tdclass">213.04</td>
                                        <td class="tdclass">37</td>
                                        <td class="tdclass">176.04</td>
                                        <td class="tdclass">0</td>
                                        <td class="tdclass">1</td>
                                    </tr>
                                </tbody>'''


def run_demo():
    recs = parse_html_to_records(HTML_SNIPPET)
    out = [r.__dict__ for r in recs]
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run_demo()
